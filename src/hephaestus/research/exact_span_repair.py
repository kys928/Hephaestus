"""A model suggests a *single textual span*, the trusted controller applies it.

This is safer and much cheaper than asking a small local coder to reproduce
the entire Python file. The model controls only one old/new text substitution;
target path, source SHA, verified failure, test suite, exact changed-lines
ceiling, sandbox and PR governance remain controller-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
from hashlib import sha256
import json

from .constrained_repair import PatchCandidate, PatchModel, RepairBoundaryError, RepairScope


@dataclass(slots=True)
class ExactSpanPatchProposer:
    model: PatchModel

    def propose(self, *, scope: RepairScope, original: bytes,
                verified_fault: dict, evidence_ref: str) -> PatchCandidate:
        scope.validate()
        if (verified_fault.get("status")!="verified_failed"
                or verified_fault.get("target_file")!=scope.allowed_file
                or not evidence_ref.startswith("sha256:")):
            raise RepairBoundaryError("single-span model repair requires reviewed failing evidence")
        if len(original)>64_000:
            raise RepairBoundaryError("single-span model source context too large")
        source=original.decode("utf-8")
        # Only a verified traceback may nominate a function to focus on.
        # It cannot broaden the code path or modify the original test.
        entrypoint=verified_fault.get("regression_entrypoint")
        if entrypoint is not None:
            if not isinstance(entrypoint,str) or not entrypoint.isidentifier():
                raise RepairBoundaryError("untrusted code entrypoint in failure evidence")
            tree=ast.parse(source)
            matching=[node for node in tree.body if isinstance(node,
                (ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==entrypoint]
            if len(matching)!=1 or matching[0].end_lineno is None:
                raise RepairBoundaryError("verified regression entrypoint is ambiguous")
            function=matching[0]
            context="".join(source.splitlines(keepends=True)[
                function.lineno-1:function.end_lineno])
        else:
            context=source
        prompt=(
            "You are a Python source repair model. Find and fix the REAL observed defect. "
            "Do not add or edit tests. Respond with ONLY one JSON object with exactly "
            "three keys: old_text, new_text, hypothesis. The old_text value must be "
            "the exact verbatim substring to replace, and must occur only once in "
            "the source. new_text is the corrected Python code for that substring. "
            "Keep the change minimal; do not rewrite the whole module. "
            "Never put Markdown fences in JSON.\n"
            "VERIFIED_FAILING_REGRESSION: "+
            json.dumps(verified_fault,sort_keys=True)+"\n"
            "SOURCE_FILE: "+scope.allowed_file+"\n"
            "SOURCE_BEGIN\n"+context+"\nSOURCE_END"
        )
        raw=self.model.generate_patch_json(prompt)
        if not isinstance(raw,str) or len(raw)>16_000:
            raise RepairBoundaryError("coder span response missing or unbounded")
        try:
            answer=json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RepairBoundaryError("coder response is not strict JSON") from exc
        if not isinstance(answer,dict) or set(answer)!={"old_text","new_text","hypothesis"}:
            raise RepairBoundaryError("coder must return exact span schema")
        before,after,hypothesis=(answer[key] for key in
                                 ("old_text","new_text","hypothesis"))
        if (not isinstance(before,str) or not isinstance(after,str)
                or not isinstance(hypothesis,str) or not 1<=len(before)<=2000
                or not 1<=len(after)<=2000 or not 10<=len(hypothesis)<=500
                or before==after or source.count(before)!=1):
            raise RepairBoundaryError("coder span ambiguous, unchanged or excessive")
        replacement=source.replace(before,after,1)
        proposed=PatchCandidate(
            path=scope.allowed_file,
            expected_source_sha256="sha256:"+sha256(original).hexdigest(),
            replacement=replacement,evidence_ref=evidence_ref,
            hypothesis=hypothesis)
        proposed.validate(scope,original)
        return proposed
