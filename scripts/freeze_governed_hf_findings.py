#!/usr/bin/env python3
"""Archive verified research findings as new release assets; never overwrite.

Invoked only after the dedicated live handoff and tests succeed. No dataset,
credentials, raw preview text or model files are added to the git tree.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

TAG = "hf-staged-review-verified-v1"
ROOT = Path("governed_hf_handoff")


class SafeArtifactRedirect(HTTPRedirectHandler):
    """Signed blob URLs authenticate themselves; do not forward GitHub auth."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("archive redirects must remain HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def verified_findings(root):
    index = json.loads((root / "sha256-index.json").read_text())
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and p.name != "sha256-index.json" and not any(v.startswith(".") for v in p.relative_to(root).parts)}
    if set(index) != actual_files or "report.json" not in index:
        raise ValueError("custody index does not cover exactly the uploaded evidence")
    for name, digest in index.items():
        path = root / name
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("evidence index escapes its archive root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("evidence file differs from its custody index")
    report = json.loads((root / "report.json").read_text())
    for field in ("new_full_campaign_execution", "corpus_access_authorized", "corpus_acquired",
                  "training_executed", "scientific_certificate_issued"):
        if report.get(field) is not False:
            raise ValueError("research freeze cannot assert acquisition, training or certification")
    steps = report["workflow_state"]["steps"]
    if (not steps["data_acquisition_audit.dataset_discovery"]["completion_marker"] or
            steps["data_acquisition_audit.dataset_selection"]["completion_marker"] or
            len(report["candidate_reviews"]) != 3):
        raise ValueError("unexpected live research finding")
    # Verify the content-addressed objects too, not just the outer ZIP index.
    for path in (root / "artifacts" / "objects" / "sha256").rglob("*"):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != path.name:
            raise ValueError("scientific object does not match its content address")
    return report


def main():
    report = verified_findings(ROOT)
    repo, commit, run = (os.environ[k] for k in ("GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID"))
    if repo != "kys928/Hephaestus" or report.get("source_sha") != commit:
        raise ValueError("release source differs from verified workflow commit")
    token = os.environ["GITHUB_TOKEN"]
    opener = build_opener(SafeArtifactRedirect())
    def request(method, url, payload=None, content_type="application/json", accept="application/vnd.github+json"):
        data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        with opener.open(Request(url, data=data, method=method, headers={
                "Authorization": "Bearer " + token, "Accept": accept,
                "Content-Type": content_type, "X-GitHub-Api-Version": "2022-11-28"}), timeout=60) as response:
            return response.read()
    api = "https://api.github.com/repos/" + repo
    expected_zip_sha = os.environ["EVIDENCE_ARTIFACT_SHA256"].removeprefix("sha256:")
    archive = request("GET", api + "/actions/artifacts/" + os.environ["EVIDENCE_ARTIFACT_ID"] + "/zip")
    if hashlib.sha256(archive).hexdigest() != expected_zip_sha:
        raise ValueError("downloaded evidence ZIP differs from uploaded artifact digest")
    manifest = {
        "schema": "hephaestus-governed-hf-findings-freeze.v1", "tag": TAG,
        "source_commit": commit, "workflow_run": int(run),
        "workflow_url": f"https://github.com/{repo}/actions/runs/{run}",
        "evidence_zip_sha256": expected_zip_sha,
        "evidence_report_sha256": hashlib.sha256((ROOT / "report.json").read_bytes()).hexdigest(),
        "candidate_reviews": report["candidate_reviews"],
        "findings": "real staged discovery persisted; no eligible source; selection refused",
        "cpu_campaign": "11/45 unchanged", "certification": "not_certified",
        "training_or_corpus_approval": False,
        "known_limitations": ["public train previews only", "source rights unresolved",
            "private frozen reference manifests and independent custody not supplied",
            "no full-corpus PII, semantic or frozen-evaluation leakage audit",
            "no new uninterrupted 45-stage campaign", "release hosting not claimed platform immutable"],
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    # A retry verifies/reconciles this exact freeze, never a different commit.
    release = None
    try:
        release = json.loads(request("GET", api + "/releases/tags/" + TAG))
    except HTTPError as exc:
        if exc.code != 404:
            raise
    if release and (release.get("target_commitish") != commit or manifest_sha not in release.get("body", "")):
        raise ValueError("findings release already exists with different evidence; refuse modification")
    if release is None:
        release = json.loads(request("POST", api + "/releases", {
        "tag_name": TAG, "target_commitish": commit,
        "name": "Governed HF staged review — verified research findings v1",
        "draft": True, "prerelease": True,
        "body": (f"Research findings at `{commit}`. [Verified run]({manifest['workflow_url']}).\n\n"
            "Three live revision-verified public previews feed persisted staged discovery and selection. "
            "Selection correctly refuses all candidates. Existing CPU campaign remains **11/45**. "
            "No corpus/training approval or scientific certificate.\n\n"
            f"Evidence ZIP SHA-256: `{expected_zip_sha}`\n\nManifest SHA-256: `{manifest_sha}`\n\n"
            "Hosting is not asserted platform-immutable; commit and byte digests are the integrity references.")
        }))
    upload = release["upload_url"].split("{")[0]
    for name, data, content_type in (
        ("governed-hf-staged-handoff.zip", archive, "application/zip"),
        ("findings-manifest.json", manifest_bytes, "application/json"),
        ("SHA256SUMS", (expected_zip_sha + "  governed-hf-staged-handoff.zip\n" +
            manifest_sha + "  findings-manifest.json\n").encode(), "text/plain")):
        matches = [a for a in release.get("assets", []) if a["name"] == name]
        if matches:
            asset = matches[0]
        elif release["draft"]:
            asset = json.loads(request("POST", upload + "?name=" + name, data, content_type))
        else:
            raise ValueError("published freeze is missing an expected asset; refuse modification")
        check = request("GET", asset["url"], accept="application/octet-stream")
        if hashlib.sha256(check).hexdigest() != hashlib.sha256(data).hexdigest():
            raise ValueError("release asset failed independent download verification")
    published = (json.loads(request("PATCH", api + "/releases/" + str(release["id"]), {"draft": False}))
                 if release["draft"] else release)
    print(json.dumps({"release": published["html_url"], "source_commit": commit,
        "manifest_sha256": manifest_sha, "evidence_zip_sha256": expected_zip_sha,
        "nonproduction_research_findings_only": True}, sort_keys=True))


if __name__ == "__main__":
    main()
