"""Private, authenticated, one-shot GPU transport bootstrap (stdlib only).

The controller supplies a digest-pinned source archive and hashed dependency
lock. No control-plane credentials, shell command endpoint or model downloads.
"""
from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
from threading import Lock, Thread

ACTIONS = ("setup", "qualify", "slot-0", "slot-1", "slot-2", "slot-3")
MAX_SOURCE = 16 * 1024 * 1024
MAX_CAPSULE = 2 * 1024 * 1024
MAX_EXTRACTED = 64 * 1024 * 1024


def extract_source(raw, destination, expected_hash):
    if len(raw) > MAX_SOURCE or sha256(raw).hexdigest() != expected_hash:
        raise ValueError("source archive differs from pinned bytes")
    destination.mkdir(mode=0o700, exist_ok=False)
    total = 0
    seen = set()
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or member.name in seen or "\\" in member.name):
                raise ValueError("unsafe or duplicate source member")
            seen.add(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError("links or special archive objects forbidden")
            if member.isfile():
                total += member.size
                if total > MAX_EXTRACTED:
                    raise ValueError("source expanded beyond ceiling")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                with target.open("xb") as output:
                    output.write(stream.read(member.size + 1))
                if target.stat().st_size != member.size:
                    raise ValueError("archive member length differs")


class Gateway:
    def __init__(self, root, token, deadline, executor):
        if not re.fullmatch(r"[a-f0-9]{64}", token) or deadline.tzinfo is None:
            raise ValueError("exact private session token and absolute deadline required")
        self.root, self.token, self.deadline, self.executor = root, token, deadline, executor
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = Lock()

    def authorized(self, value):
        return compare_digest(value or "", "Bearer " + self.token)

    def status(self, name):
        if name not in ACTIONS:
            raise ValueError("unknown operation")
        terminal = self.root / (name + ".result.json")
        if terminal.exists():
            return json.loads(terminal.read_bytes())
        return {"status": "running" if (self.root / (name + ".consumed")).exists() else "not_started"}

    def start(self, name, raw):
        if name not in ACTIONS or datetime.now(timezone.utc) >= self.deadline:
            raise PermissionError("operation outside bounded mission")
        limit = MAX_SOURCE if name == "setup" else MAX_CAPSULE
        if len(raw) > limit:
            raise ValueError("operation exceeds payload ceiling")
        if name == "qualify" and raw != b"{}":
            raise ValueError("qualify does not accept a command")
        if name.startswith("slot-"):
            capsule = json.loads(raw)
            if capsule.get("slot") != int(name[-1]) or capsule.get("schema") != "anvil.worker-capsule.v1":
                raise ValueError("capsule differs from exact slot")
        with self.lock:
            if self.status(name)["status"] != "not_started":
                return False
            if any(self.status(previous)["status"] != "complete" for previous in ACTIONS[:ACTIONS.index(name)]):
                raise PermissionError("predecessor operation is incomplete")
            # Consume before launching; a crash or lost HTTP acknowledgement
            # cannot turn into another inference.
            with (self.root / (name + ".consumed")).open("xb") as marker:
                marker.write(sha256(raw).hexdigest().encode())
                marker.flush()
                os.fsync(marker.fileno())
            Thread(target=self._run, args=(name, raw), daemon=True).start()
            return True

    def _run(self, name, raw):
        try:
            result = {"status": "complete", "result": self.executor(name, raw)}
        except Exception as exc:
            result = {"status": "failed", "error_type": type(exc).__name__}
        encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > MAX_CAPSULE:
            encoded = b'{"status":"failed","error_type":"ResultTooLarge"}'
        tmp = self.root / (name + ".result.tmp")
        with tmp.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, self.root / (name + ".result.json"))


def command_executor(root, deadline, source_hash, lock_hash):
    source = root / "source"

    def run(args, raw=None, runtime_ref=None):
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("execution deadline reached")
        env = {k: v for k, v in os.environ.items() if k not in {
            "ANVIL_LIVE_TOKEN", "RUNPOD_API_KEY", "ANVIL_POSTGRES_DSN",
            "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY", "HF_TOKEN", "GITHUB_TOKEN",
        }}
        env.update(PYTHONPATH=str(source / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if runtime_ref is not None:
            env["ANVIL_RUNTIME_REF"] = runtime_ref
        result = subprocess.run(args, cwd=source, env=env, input=raw, capture_output=True,
                                timeout=min(remaining, 600), check=False)
        if result.returncode or len(result.stdout) > MAX_CAPSULE:
            raise RuntimeError("bounded worker command failed")
        return result.stdout.decode("utf-8")

    def execute(name, raw):
        if name == "setup":
            extract_source(raw, source, source_hash)
            lock = source / "requirements-anvil-live.lock"
            if not lock.is_file() or sha256(lock.read_bytes()).hexdigest() != lock_hash:
                raise PermissionError("reviewed dependency lock absent or altered")
            run([sys.executable, "-m", "pip", "install", "--require-hashes", "--no-deps",
                 "--only-binary=:all:", "-r", str(lock)])
            run([sys.executable, "-m", "pip", "check"])
            return {"source_sha256": source_hash, "lock_sha256": lock_hash}
        if name == "qualify":
            out = run([sys.executable, "-u", "scripts/anvil_qualify_cuda_offline.py"])
            rows = []
            for line in out.splitlines():
                if line.startswith("{"):
                    doc = json.loads(line)
                    if doc.get("schema") == "anvil.cuda-offline-load.v1":
                        rows.append(doc)
            if [r["role"] for r in rows] != ["diagnosis", "planner"]:
                raise RuntimeError("missing exact two-role load evidence")
            runtime = {"schema": "anvil.qualified-runtime.v1", "pod_id": os.environ["RUNPOD_POD_ID"],
                       "source_sha256": source_hash, "lock_sha256": lock_hash, "loads": rows}
            ref = "sha256:" + sha256(json.dumps(runtime, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            return {"runtime": runtime, "runtime_ref": ref}
        qualified = json.loads((root / "qualify.result.json").read_bytes())
        if qualified.get("status") != "complete":
            raise PermissionError("runtime qualification absent")
        runtime_ref = qualified["result"]["runtime_ref"]
        if json.loads(raw).get("runtime_ref") != runtime_ref:
            raise PermissionError("slot differs from actual worker load evidence")
        out = run([sys.executable, "-u", "scripts/anvil_live_worker.py"], raw, runtime_ref=runtime_ref)
        rows = [json.loads(line.removeprefix("ANVIL_LIVE_RECEIPT=")) for line in out.splitlines()
                if line.startswith("ANVIL_LIVE_RECEIPT=")]
        if len(rows) != 1:
            raise RuntimeError("missing unique actual worker receipt")
        return rows[0]
    return execute


def serve(gateway):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def reply(self, code, doc):
            data = json.dumps(doc, sort_keys=True, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def check(self):
            if not gateway.authorized(self.headers.get("Authorization")):
                self.reply(403, {"status": "denied"})
                return False
            return True

        def do_GET(self):
            if not self.check():
                return
            if self.path == "/health":
                self.reply(200, {"status": "ready"})
            elif self.path.startswith("/result/") and self.path[8:] in ACTIONS:
                self.reply(200, gateway.status(self.path[8:]))
            else:
                self.reply(404, {"status": "unknown"})

        def do_POST(self):
            if not self.check():
                return
            name = self.path.removeprefix("/execute/")
            if self.path != "/execute/" + name or name not in ACTIONS:
                self.reply(404, {"status": "unknown"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                limit = MAX_SOURCE if name == "setup" else MAX_CAPSULE
                if not 0 <= size <= limit:
                    raise ValueError("payload length")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("truncated request")
                started = gateway.start(name, raw)
                self.reply(202 if started else 409, {"status": "consumed"})
            except (ValueError, PermissionError):
                self.reply(400, {"status": "rejected"})

    server = ThreadingHTTPServer(("0.0.0.0", 8787), Handler)
    server.timeout = 1
    try:
        while datetime.now(timezone.utc) < gateway.deadline:
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    root = Path("/tmp/anvil-private-gateway")
    deadline = datetime.fromisoformat(os.environ["ANVIL_DEADLINE_UTC"])
    gateway = Gateway(root, os.environ["ANVIL_LIVE_TOKEN"], deadline,
                      command_executor(root, deadline, os.environ["ANVIL_SOURCE_SHA256"],
                                       os.environ["ANVIL_PACKAGE_LOCK_SHA256"]))
    serve(gateway)
