"""Public rehearsal service fixture. Never an independent custodian or final grant."""
from datetime import datetime,timezone
from hephaestus.schemas.exposure_ledger import ExposureReservation, ExposureReceipt
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.verified import canonical_bytes


class FixtureExposureLedger:
    authority_scope='nonconfirmatory-fixture'

    def __init__(self,repository,*,private_key,authority_id,trusted_custodian_keys,namespace):
        if not namespace.startswith('fixture/'):
            raise ValueError('fixture ledger namespace must be explicit')
        self.repository=repository
        self.key,self.authority_id=private_key,authority_id
        self.trusted_custodian_keys=trusted_custodian_keys
        self.namespace=namespace

    def consume_once(self,reservation,grant,*,submission):
        from hephaestus.evaluation.prospective_pipeline import digest,verify_custodian_grant
        record=ExposureReservation.from_dict(reservation)
        if record.purpose!='rehearsal':
            raise PermissionError('fixture ledger cannot authorize protected final exposure')
        if (digest(submission)!=record.submission_commitment
                or submission.get('holdout_commitment')!=record.material_commitment
                or submission.get('study_id')!=record.study_id):
            raise PermissionError('reservation material is not bound by the signed submission')
        verify_custodian_grant(grant,trusted_public_keys=self.trusted_custodian_keys,expected={
            'study_id':record.study_id,'purpose':record.purpose,'allowed_sessions':1,
            'submission_commitment':record.submission_commitment})
        if digest(grant)!=record.grant_ref:raise PermissionError('reservation grant identity differs')
        with execution_owner(self.repository,'fixture-exposure-ledger',self.namespace):
            previous=self.repository.all('fixture_exposure_consumptions')
            if any(row['material_commitment']==record.material_commitment
                    or row['reservation']['grant_ref']==record.grant_ref
                    or row['reservation']['study_id']==record.study_id for row in previous):
                raise PermissionError('external fixture ledger: material permanently consumed; no retry or permission reissue')
            receipt=ExposureReceipt(self.authority_id,self.namespace,self.authority_scope,
                digest(reservation),record.material_commitment,len(previous)+1,datetime.now(timezone.utc).isoformat()).to_dict()
            envelope={'payload':receipt,'signature_hex':self.key.sign(canonical_bytes(receipt)).hex()}
            self.repository.append('fixture_exposure_consumptions',{'material_commitment':record.material_commitment,
                'reservation':reservation,'receipt':envelope},operation_id='exposure-'+record.material_commitment[7:])
            return envelope
