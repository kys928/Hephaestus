from dataclasses import dataclass
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from datetime import datetime,timezone


@dataclass(slots=True)
class CpuAutonomyStore:
    repository: object
    artifacts: object

    def freeze(self, benchmark_id, manifest):
        ref = self.artifacts.put_bytes(canonical_bytes(manifest)).artifact_ref
        old = self.repository.get_latest("cpu_autonomy_manifests", "benchmark_id", benchmark_id)
        if old and old["manifest_ref"] != ref:
            raise PermissionError("CPU benchmark manifest cannot change after freeze")
        if not old:
            self.repository.append("cpu_autonomy_manifests", {"benchmark_id":benchmark_id,"manifest_ref":ref})
        return ref

    def final(self, program_id):
        return self.repository.get_latest("cpu_autonomy_final", "program_id", program_id)

    def begin_program(self,program_id,identity):
        old=self.repository.get_latest("cpu_autonomy_programs","program_id",program_id)
        if old and old["identity"]!=identity:
            raise PermissionError("CPU program policy/seed/manifest changed across restart")
        if not old:
            old={"program_id":program_id,"identity":identity,"started_at":datetime.now(timezone.utc).isoformat()}
            self.repository.append("cpu_autonomy_programs",old)
        return old

    def consume_final(self, program_id, identity):
        if self.final(program_id):
            raise PermissionError("final CPU benchmark assessment already consumed")
        self.repository.append("cpu_autonomy_final", {"program_id":program_id,"identity":identity,"status":"consumed"})

    def complete_final(self, program_id, identity, result):
        old = self.final(program_id)
        if old is None or old["status"] != "consumed" or old["identity"] != identity:
            raise PermissionError("CPU final result has no matching exposure intent")
        ref = self.artifacts.put_bytes(canonical_bytes(result)).artifact_ref
        self.repository.append("cpu_autonomy_final", {**old,"status":"completed","result_ref":ref})
        return read_verified_json(self.artifacts,ref)
