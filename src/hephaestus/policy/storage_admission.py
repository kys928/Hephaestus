"""Fail-closed Linux single-host path/mount admission before storage mutation."""
import os
from pathlib import Path
import re
import stat
import sys

from hephaestus.schemas.storage_admission import LocalStorageProfile, StorageAdmission


class UnsupportedStorage(PermissionError):
    pass


def reject_symlinks(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts:
        raise UnsupportedStorage("storage path must not contain parent traversal")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise UnsupportedStorage("symlinked storage paths are unsupported")
    return path


def _mounts():
    def decode(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1],8)), value)
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields=line.split(); split=fields.index('-')
        mounts.append((decode(fields[4]), fields[split+1], fields[2]))
    return mounts


def admit_storage(path: Path, profile: LocalStorageProfile | None = None) -> StorageAdmission:
    profile = profile or LocalStorageProfile()
    profile.__post_init__()
    if sys.platform != 'linux':
        raise UnsupportedStorage("storage profile requires Linux process-safe locking and directory fsync")
    root = reject_symlinks(Path(profile.root))
    path = reject_symlinks(Path(path))
    permitted_roots=(Path('/tmp'),Path('/var/tmp'),Path('/var/lib/hephaestus'))
    if (not any(root.is_relative_to(p) for p in permitted_roots)
            or not path.is_relative_to(root)):
        raise UnsupportedStorage("storage path is outside the admitted isolated local root")
    existing = path
    while not existing.exists():
        existing=existing.parent
    for item in (existing, *existing.parents):
        if not item.is_relative_to(root):
            break
        metadata=item.stat()
        if metadata.st_uid != os.geteuid() and item not in permitted_roots:
            raise UnsupportedStorage("storage entries must be owned by this deployment user")
        if metadata.st_mode & (stat.S_IWGRP|stat.S_IWOTH) and not (
                item in permitted_roots and metadata.st_mode & stat.S_ISVTX):
            raise UnsupportedStorage("storage entries must not be writable by other users")
    candidates=[r for r in _mounts() if existing.is_relative_to(Path(r[0]))]
    if not candidates:
        raise UnsupportedStorage("storage mount could not be identified")
    mountpoint,filesystem,device=max(candidates,key=lambda row:len(Path(row[0]).parts))
    supported={'ext4','xfs','btrfs'}
    if profile.purpose=='bounded_rehearsal':
        supported.add('overlay')
    if filesystem not in supported:
        raise UnsupportedStorage(f"storage filesystem {filesystem!r} is not admitted for {profile.purpose}")
    if profile.purpose=='persistent_research' and any(root.is_relative_to(p) for p in permitted_roots[:2]):
        raise UnsupportedStorage("persistent research storage requires an operator-managed retained root")
    return StorageAdmission(profile.profile_id,profile.purpose,str(path),str(root),filesystem,
        mountpoint,device, 'container/host-lifetime-only' if profile.purpose=='bounded_rehearsal'
        else 'operator-managed-host-local-fsync')
