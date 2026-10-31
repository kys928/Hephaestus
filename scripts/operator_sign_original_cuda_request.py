#!/usr/bin/env python3
"""External operator utility: sign a reviewed GPU compute request OFF-worker.

The private key is supplied by the HUMAN OPERATOR as a local, owner-only file.
Never generate a key here, never echo it, never store it in the repository,
and never allow the GPU worker or a training model to call this program.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json,os,stat
from pathlib import Path

from hephaestus.storage.verified import canonical_bytes
from hephaestus.production.signed_cuda_grants import expected_gpu_operation_map


def sign_request(request_file, *, private_key_file, operator_id, expires_at, output):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    path=Path(private_key_file)
    info=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or (info.st_mode & 0o077)
        or hasattr(os,"getuid") and info.st_uid!=os.getuid()):
        raise PermissionError("external owner key requires a regular owner-only file")
    if (not isinstance(operator_id,str) or operator_id in ("","self","model",
         "REQUIRES_EXTERNAL_OPERATOR_ID") or len(operator_id)>128):
        raise PermissionError("a real independent operator identity is required")
    until=datetime.fromisoformat(expires_at.replace("Z","+00:00"))
    if until.tzinfo is None or until<=datetime.now(timezone.utc):
        raise PermissionError("signed GPU authorization must expire in future UTC")
    request=json.loads(Path(request_file).read_bytes())
    if (request.get("status")!="requested"
        or request.get("schema")!="tier-a-original-cuda-compute-authorization.v1"
        or request.get("operations")!=expected_gpu_operation_map(request.get("workflow_id"))
        or request.get("device")!="cuda:0"
        or request.get("external_certification") is not False):
        raise PermissionError("operator can only sign exact reviewed pending CUDA request")
    auth={**request,"operator_id":operator_id,"expires_at":until.isoformat(),"status":"approved"}
    private_bytes=bytes.fromhex(path.read_text(encoding="ascii").strip())
    if len(private_bytes)!=32:
        raise PermissionError("external signer needs 32-byte Ed25519 private seed")
    private=Ed25519PrivateKey.from_private_bytes(private_bytes)
    public=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    envelope={"authorization":auth,"signature_hex":private.sign(canonical_bytes(auth)).hex()}
    output=Path(output)
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(canonical_bytes(envelope))
            stream.flush();os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
      "scope":"original_internal_generated_data_cuda_only",
      "signed_file_sha256":"sha256:"+sha256(output.read_bytes()).hexdigest(),
      "trusted_public_key_hex":public.hex(),
      "trusted_public_key_sha256":"sha256:"+sha256(public).hexdigest(),
      "operator_id":operator_id,"expires_at":auth["expires_at"],
      "gpu_authorized":True,"independent_scientific_certification":False,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--private-key-file",type=Path,required=True)
    parser.add_argument("--operator-id",required=True)
    parser.add_argument("--expires-at",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(sign_request(args.request,
        private_key_file=args.private_key_file,
        operator_id=args.operator_id,
        expires_at=args.expires_at,
        output=args.output),sort_keys=True))

if __name__=="__main__":
    main()
