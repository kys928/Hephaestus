"""Retain fixed public measurements without duplicating acknowledged work.

Only development-bucket, nonconfirmatory pure generation is retryable here.
This is never a final exposure ledger or optimizer restart mechanism.
"""
from dataclasses import dataclass
from hephaestus.storage.verified import canonical_bytes,read_verified_json
from .execution_ownership import execution_owner


@dataclass(slots=True)
class PublicPilotGenerationStore:
    repository: object
    artifacts: object

    def measure(self,request,producer):
        from hephaestus.evaluation.prospective_pipeline import digest
        request.__post_init__()
        key=digest(request.to_dict())[7:]
        if self.repository._db.witness() is None:
            raise PermissionError('public measurement restart requires an enrolled campaign witness')
        with execution_owner(self.repository,'public-pilot-measurement',key):
            prior=self.repository.latest_by_key('public_pilot_measurements',key)
            if prior is not None:
                if prior['request']!=request.to_dict():raise PermissionError('retained public measurement scope differs')
                return read_verified_json(self.artifacts,prior['raw_sample_ref'])
            sample=producer()
            if sample.get('sample_id')!=request.sample_id:
                raise ValueError('public generator returned another canonical sample')
            reference=self.artifacts.put_bytes(canonical_bytes(sample)).artifact_ref
            self.repository.append('public_pilot_measurements',{
                'request':request.to_dict(),'raw_sample_ref':reference},
                operation_id='public-measurement-'+key,record_key=key)
            return sample
