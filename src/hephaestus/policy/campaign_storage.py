"""A campaign backup must not contain its rollback witness or kernel fence."""
import hashlib
import os
from pathlib import Path
from hephaestus.schemas.storage_admission import LocalStorageProfile
from .storage_admission import admit_storage, reject_symlinks


def campaign_control_path(database: Path, profile: LocalStorageProfile) -> Path:
    base = (Path('/var/lib/hephaestus/control') if profile.purpose == 'persistent_research'
            else Path('/tmp') / f'hephaestus-campaign-control-{os.geteuid()}')
    # The control root is deliberately outside even a narrowly admitted campaign
    # root, but obeys the same mount/purpose policy.
    admit_storage(base, LocalStorageProfile(
        root='/var/lib/hephaestus' if profile.purpose == 'persistent_research' else '/tmp',
        purpose=profile.purpose))
    return base / hashlib.sha256(str(reject_symlinks(database)).encode()).hexdigest()


def require_separate_control(campaign_root: Path, control: Path):
    root, control = reject_symlinks(campaign_root), reject_symlinks(control)
    if root.is_relative_to(control) or control.is_relative_to(root):
        raise PermissionError('campaign backup and retained witness must be disjoint directories')
