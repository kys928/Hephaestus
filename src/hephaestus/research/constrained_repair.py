"""Sandbox-only bounded self-repair for a narrow diagnosis-code slice.

Model-generated edits are untrusted *proposals*. No model holds authority to run
shell commands, modify a live checkout, change policies, mutate eval packs,
approve its own patch, or merge a branch. A trusted publisher may create a
reviewable PR only after real baseline/candidate sandbox receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol
import base64
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


_SOURCE_PREFIX = "src/hephaestus/diagnosis/"
_TESTS = frozenset({
    "tests/test_targeted_post_failure_probes.py",
    "tests/test_evidence_based_diagnosis.py",
})
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN = frozenset({".git", ".env", ".env.local", "secrets", "private",
                        "state", "artifacts", "checkpoints", "model_weights", "runs"})


class RepairBoundaryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RepairScope:
    """Operator-controlled scope; never sourced from the model response."""

    base_commit: str
    allowed_file: str
    fixed_test: str
    maximum_changed_lines: int = 100

    def validate(self) -> None:
        if not _COMMIT.fullmatch(self.base_commit):
            raise RepairBoundaryError("exact approved Git commit is required")
        path = PurePosixPath(self.allowed_file)
        if (self.allowed_file != path.as_posix() or not self.allowed_file.startswith(_SOURCE_PREFIX)
                or path.suffix != ".py" or ".." in path.parts
                or path.name.startswith("_") or not path.name):
            raise RepairBoundaryError("only one approved diagnosis Python module is repairable")
        if self.fixed_test not in _TESTS:
            raise RepairBoundaryError("test command must use a preserved review-approved suite")
        if not 1 <= self.maximum_changed_lines <= 150:
            raise RepairBoundaryError("code change exceeds the bounded repair ceiling")


@dataclass(frozen=True, slots=True)
class PatchCandidate:
    path: str
    expected_source_sha256: str
    replacement: str
    evidence_ref: str
    hypothesis: str

    def validate(self, scope: RepairScope, original: bytes) -> tuple[str, int]:
        scope.validate()
        if self.path != scope.allowed_file or not _HASH.fullmatch(self.evidence_ref):
            raise RepairBoundaryError("patch outside operator-controlled scope or without evidence")
        if (not _HASH.fullmatch(self.expected_source_sha256)
                or "sha256:" + sha256(original).hexdigest() != self.expected_source_sha256):
            raise RepairBoundaryError("repair proposal uses stale or fabricated source bytes")
        if (not isinstance(self.replacement, str) or not 10 <= len(self.replacement.encode()) <= 64_000
                or not 10 <= len(self.hypothesis) <= 500 or "\x00" in self.replacement):
            raise RepairBoundaryError("patch size or hypothesis invalid")
        try:
            compile(self.replacement, self.path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            raise RepairBoundaryError("replacement does not parse as Python") from exc
        if self.replacement.encode() == original:
            raise RepairBoundaryError("candidate patch changes no bytes")
        old_lines = original.decode("utf-8").splitlines()
        new_lines = self.replacement.splitlines()
        diff = list(difflib.ndiff(old_lines, new_lines))
        touched = sum(line.startswith(("+ ", "- ")) for line in diff)
        if touched > scope.maximum_changed_lines:
            raise RepairBoundaryError("patch changes too many lines")
        return "sha256:" + sha256(self.replacement.encode()).hexdigest(), touched


class PatchModel(Protocol):
    def generate_patch_json(self, prompt: str) -> str: ...


@dataclass(slots=True)
class BoundedPatchProposer:
    """A model can supply replacement text but cannot select target or tests."""

    model: PatchModel

    def propose(self, *, scope: RepairScope, original: bytes,
                verified_fault: Mapping[str, object], evidence_ref: str) -> PatchCandidate:
        scope.validate()
        if (not _HASH.fullmatch(evidence_ref) or verified_fault.get("status") != "verified_failed"
                or verified_fault.get("target_file") != scope.allowed_file):
            raise RepairBoundaryError("no independently verified in-scope failing regression")
        if len(original) > 64_000:
            raise RepairBoundaryError("code source exceeds fixed model context")
        prompt = (
            "Generate one bounded source edit as a JSON object with exactly keys "
            "path,expected_source_sha256,replacement,evidence_ref,hypothesis. "
            "Do not edit tests, policies, evaluation packs or any other file. "
            "Do not include commands or markdown. Verification occurs in a "
            "network-disabled sandbox.\n"
            "APPROVED_PATH=" + scope.allowed_file + "\n"
            "EXPECTED_SHA256=sha256:" + sha256(original).hexdigest() + "\n"
            "EVIDENCE_REF=" + evidence_ref + "\n"
            "VERIFIED_FAULT=" + json.dumps(verified_fault, sort_keys=True) + "\n"
            "SOURCE_START\n" + original.decode("utf-8") + "\nSOURCE_END"
        )
        raw = self.model.generate_patch_json(prompt)
        if not isinstance(raw, str) or len(raw) > 100_000:
            raise RepairBoundaryError("model patch response not bounded text")
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RepairBoundaryError("model patch response is not one JSON object") from exc
        if not isinstance(doc, dict) or set(doc) != {
                "path","expected_source_sha256","replacement","evidence_ref","hypothesis"}:
            raise RepairBoundaryError("model patch has unexpected fields")
        candidate = PatchCandidate(**doc)
        candidate.validate(scope, original)
        return candidate


@dataclass(frozen=True, slots=True)
class SandboxReceipt:
    status: str
    exit_code: int
    output_sha256: str
    output_excerpt: str
    worker: str


class SandboxRunner(Protocol):
    def run(self, checkout: Path, *, fixed_test: str) -> SandboxReceipt: ...


@dataclass(frozen=True, slots=True)
class OfflineDockerPytestRunner:
    """Execute potentially hostile candidate tests in an unprivileged container.

    The image must already exist and be *digest-pinned*. No secret values,
    Docker socket, host home, network, GPU, or writable source are mounted.
    This is a Linux Docker boundary; Windows/Pod deployment needs an
    independently accepted equivalent runner.
    """

    image_digest: str
    docker_executable: str = "/usr/bin/docker"
    timeout_seconds: int = 240

    def run(self, checkout: Path, *, fixed_test: str) -> SandboxReceipt:
        if fixed_test not in _TESTS:
            raise RepairBoundaryError("test suite changed outside operator-approved scope")
        if not (re.fullmatch(r"[A-Za-z0-9_./:-]+@sha256:[0-9a-f]{64}", self.image_digest)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest)):
            raise RepairBoundaryError("sandbox runtime must use a reviewed image digest")
        if not Path(self.docker_executable).is_file():
            raise RepairBoundaryError("reviewed Docker executable not installed")
        cmd = [
            self.docker_executable, "run", "--rm", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "128", "--memory", "1024m", "--cpus", "2",
            "--read-only", "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--mount", f"type=bind,src={checkout.resolve()},dst=/work,readonly",
            "--workdir", "/work", "--env", "PYTHONPATH=/work/src",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            self.image_digest, "python", "-B", "-m", "pytest", "-q",
            "-p", "no:cacheprovider", fixed_test,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                timeout=self.timeout_seconds, check=False, env={
                    "PATH":"/usr/bin:/bin", "HOME":"/nonexistent",
                })
        except subprocess.TimeoutExpired:
            return SandboxReceipt("timeout",124,"sha256:"+sha256(b"timeout").hexdigest(),
                                  "", "offline-docker-pytest.v1")
        output = (result.stdout + "\n" + result.stderr).encode("utf-8",errors="replace")
        # Never append full logs to operator-facing control records.
        return SandboxReceipt("passed" if result.returncode == 0 else "failed",
              result.returncode, "sha256:"+sha256(output).hexdigest(),
              output.decode(errors="replace")[-2500:], "offline-docker-pytest.v1")


def _prepare_copy(source_root: Path, target: Path) -> None:
    """Copy only reviewable code/tests/configs; never .git, data or secrets."""
    if not (source_root/"src"/"hephaestus").is_dir() or not (source_root/"tests").is_dir():
        raise RepairBoundaryError("no complete source/test workspace")
    for directory in ("src","tests","configs","hephaestus"):
        root = source_root / directory
        if not root.exists():
            continue
        for walk_root, dirs, files in os.walk(root, followlinks=False):
            current=Path(walk_root)
            if current.is_symlink():
                raise RepairBoundaryError("symlink inside repair source checkout")
            if any((current/item).is_symlink() for item in dirs):
                raise RepairBoundaryError("symlink inside repair source checkout")
            dirs[:] = [item for item in dirs if item not in _FORBIDDEN and
                      not item.startswith(".") and item != "__pycache__"]
            for name in files:
                path=current/name
                if path.is_symlink():
                    raise RepairBoundaryError("symlink inside repair source checkout")
                if name in _FORBIDDEN or name.startswith("."):
                    continue
                dest = target / path.relative_to(source_root)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
    setup=source_root/"pyproject.toml"
    if setup.is_file() and not setup.is_symlink():
        shutil.copyfile(setup,target/"pyproject.toml")


class GitHubPatchPublisher(Protocol):
    def publish(self, *, scope: RepairScope, candidate: PatchCandidate,
                verified_trial: Mapping[str, object]) -> str: ...


@dataclass(slots=True)
class ConstrainedSelfRepair:
    """Execute red/green evidence-only repair, then optionally publish review PR."""

    artifacts: ArtifactStore
    runner: SandboxRunner

    def trial(self, *, checkout: Path, scope: RepairScope, candidate: PatchCandidate,
              verified_fault_ref: str) -> dict[str, object]:
        scope.validate()
        if not _HASH.fullmatch(verified_fault_ref) or verified_fault_ref != candidate.evidence_ref:
            raise RepairBoundaryError("repair candidate is not bound to the verified failure")
        fact = read_verified_json(self.artifacts, verified_fault_ref)
        if (fact.get("schema") != "reviewed-code-failure.v1"
                or fact.get("status") != "verified_failed"
                or fact.get("target_file") != scope.allowed_file):
            raise RepairBoundaryError("repair source evidence not independently admitted")
        base = Path(checkout)
        path = base/scope.allowed_file
        if path.is_symlink() or not path.is_file():
            raise RepairBoundaryError("approved source path unavailable or symlink")
        original = path.read_bytes()
        # The designated source checkout MUST be the actually reviewed commit,
        # not just a directory whose one selected file happens to have a hash.
        head = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=10)
        if head.returncode != 0 or head.stdout.strip() != scope.base_commit:
            raise RepairBoundaryError("source checkout does not match authorized Git commit")
        status = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=False, timeout=10)
        if status.returncode != 0 or status.stdout.strip():
            raise RepairBoundaryError("source checkout is dirty before repair")
        new_sha,changed_lines = candidate.validate(scope,original)
        with tempfile.TemporaryDirectory(prefix="hephaestus-untrusted-patch-") as folder:
            stage=Path(folder)
            clean=stage/"clean"
            changed=stage/"candidate"
            clean.mkdir()
            changed.mkdir()
            _prepare_copy(base,clean)
            _prepare_copy(base,changed)
            target=changed/scope.allowed_file
            target.write_text(candidate.replacement,encoding="utf-8")
            baseline=self.runner.run(clean, fixed_test=scope.fixed_test)
            repaired=self.runner.run(changed,fixed_test=scope.fixed_test)
        # Existing tests MUST first reproduce the reported failure and then
        # pass on the changed code. This is an observed regression repair,
        # not evidence of model-level scientific improvement.
        accepted=(baseline.status=="failed" and repaired.status=="passed"
                  and baseline.worker==repaired.worker
                  and baseline.worker=="offline-docker-pytest.v1")
        report={
            "schema":"constrained-self-repair-trial.v1",
            "base_commit":scope.base_commit,
            "target_file":scope.allowed_file,
            "fixed_test":scope.fixed_test,
            "expected_source_sha256":candidate.expected_source_sha256,
            "replacement_sha256":new_sha,
            "changed_lines":changed_lines,
            "verified_fault_ref":verified_fault_ref,
            "baseline":baseline.__dict__ if hasattr(baseline,"__dict__") else {
                name:getattr(baseline,name) for name in ("status","exit_code","output_sha256","output_excerpt","worker")},
            "candidate":{
                name:getattr(repaired,name) for name in ("status","exit_code","output_sha256","output_excerpt","worker")},
            "red_to_green_verified":accepted,
            "production_source_mutated":False,
            "model_weights_mutated":False,
            "frozen_eval_mutated":False,
            "automatic_merge_authorized":False,
            "status":"reviewable_patch" if accepted else "rejected_patch",
        }
        artifact=self.artifacts.put_bytes(canonical_bytes(report),media_type="application/json")
        return {**report,"trial_ref":artifact.artifact_ref}

    def submit_for_review(self, *, scope: RepairScope, candidate: PatchCandidate,
                          trial: Mapping[str, object], publisher: GitHubPatchPublisher) -> str:
        # A publisher can create only a review branch/PR under independent GitHub
        # permission; it cannot merge, activate, or change the frozen evaluator.
        ref=trial.get("trial_ref")
        if (not isinstance(ref,str) or not _HASH.fullmatch(ref)
                or trial.get("red_to_green_verified") is not True
                or trial.get("status") != "reviewable_patch"
                or trial.get("base_commit") != scope.base_commit
                or trial.get("target_file") != candidate.path
                or trial.get("replacement_sha256") !=
                    "sha256:"+sha256(candidate.replacement.encode()).hexdigest()):
            raise RepairBoundaryError("patch has no accepted immutable sandbox trial")
        sealed=read_verified_json(self.artifacts,ref)
        if {key:value for key,value in trial.items() if key!="trial_ref"} != sealed:
            raise RepairBoundaryError("sandbox acceptance receipt changed after verification")
        return publisher.publish(scope=scope,candidate=candidate,verified_trial=trial)


@dataclass(slots=True)
class GitHubReviewBranchPublisher:
    """Publish a bounded patch as a new review PR, never merge it.

    Connection uses a dedicated, narrowly scoped GitHub token provided by the
    trusted deployment. The model sees neither token nor arbitrary REST verbs.
    An immutable main-head comparison prevents publishing against changed
    source after the sandbox trial.
    """

    owner: str
    repository: str
    token: str
    timeout_seconds: int = 20

    def _request(self, method: str, route: str, body: dict | None = None) -> dict:
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+",self.owner)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+",self.repository)
                or not self.token or method not in {"GET","POST","PUT"}
                or not route.startswith("/repos/"+self.owner+"/"+self.repository+"/")
                or ".." in route):
            raise RepairBoundaryError("invalid dedicated review repository authority")
        url="https://api.github.com"+route
        data=canonical_bytes(body) if body is not None else None
        headers={
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
            "User-Agent":"Hephaestus-Bounded-Repair",
            "Authorization":"Bearer "+self.token,
        }
        if data is not None:
            headers["Content-Type"]="application/json"
        request=urllib.request.Request(url,data=data,headers=headers,method=method)
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_seconds) as response:
                value=json.loads(response.read(1_000_001))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            # The URL, body and credentials never enter the exception message
            # or public report. The branch may exist after an interrupted POST;
            # reconciliation is required instead of blind retry.
            raise RepairBoundaryError(
                "review-publisher-remote-operation-uncertain:"+type(exc).__name__) from None
        if not isinstance(value,dict):
            raise RepairBoundaryError("invalid GitHub reviewer response")
        return value

    def publish(self, *, scope: RepairScope, candidate: PatchCandidate,
                verified_trial: Mapping[str, object]) -> str:
        scope.validate()
        if (candidate.path != scope.allowed_file or
                verified_trial.get("status") != "reviewable_patch" or
                verified_trial.get("red_to_green_verified") is not True):
            raise RepairBoundaryError("review publisher received an unapproved patch")
        prefix=f"/repos/{self.owner}/{self.repository}"
        head=self._request("GET",prefix+"/git/ref/heads/main")
        current=(head.get("object") or {}).get("sha")
        if current != scope.base_commit:
            raise RepairBoundaryError("main advanced after original repair checkout")
        branch="repair/diagnosis-"+str(verified_trial["replacement_sha256"])[7:23]
        # Only GitHub can create this branch; no arbitrary git push to main.
        self._request("POST",prefix+"/git/refs",{
            "ref":"refs/heads/"+branch,"sha":scope.base_commit})
        response=self._request("GET",prefix+"/contents/"+scope.allowed_file+"?ref="+branch)
        source=base64.b64decode(str(response.get("content","")),validate=False)
        if ("sha256:"+sha256(source).hexdigest()!=candidate.expected_source_sha256
                or not response.get("sha")):
            raise RepairBoundaryError("GitHub candidate branch source differed from reviewed source")
        self._request("PUT",prefix+"/contents/"+scope.allowed_file,{
            "message":"repair(diagnosis): proposed bounded red-to-green diagnostic fix",
            "branch":branch,"sha":response["sha"],
            "content":base64.b64encode(candidate.replacement.encode()).decode("ascii")})
        result=self._request("POST",prefix+"/pulls",{
            "title":"Bounded Diagnostician self-repair: "+scope.allowed_file.split("/")[-1],
            "head":branch,"base":"main","draft":True,
            "body":"Automated sandbox-validated source repair ONLY; not merged or production-certified.\\n\\n"
                   "Verified failure: "+candidate.evidence_ref+"\\n"
                   "Sandbox trial: "+str(verified_trial["trial_ref"])+"\\n"
                   "Original pinned commit: "+scope.base_commit+"\\n"
                   "Fixed regression: "+scope.fixed_test+"\\n"
                   "Red -> green: true. Manual independent review required before any merge."})
        url=str(result.get("html_url") or "")
        if not re.fullmatch(r"https://github\\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+",url):
            raise RepairBoundaryError("GitHub review PR result did not return its canonical URL")
        return url
