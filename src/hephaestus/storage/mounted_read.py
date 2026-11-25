"""Read regular files under an admitted mount without following symlinks."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping


@dataclass(frozen=True, slots=True)
class MountedReadOnlyObjectSource:
    mount_root: Path
    expected_bucket: str
    require_mount: bool = False
    object_paths: Mapping[str, str] | None = None

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BinaryIO]:
        if Bucket != self.expected_bucket or not isinstance(Key, str):
            raise PermissionError("mounted volume identity differs")
        relative = Key
        if self.object_paths is not None:
            try:
                relative = self.object_paths[Key]
            except KeyError as exc:
                raise PermissionError("unregistered mounted object") from exc
        return {"Body": self.open_file(relative)}

    def open_file(self, relative: str) -> BinaryIO:
        segments = relative.split("/")
        if (
            any(segment in {"", ".", ".."} for segment in segments)
            or "\\" in relative
            or "\x00" in relative
        ):
            raise PermissionError("invalid mounted object path")
        try:
            return self._open_segments(segments)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise PermissionError("mounted object path contains a symlink or non-directory") from exc
            raise

    def _open_segments(self, segments: list[str]) -> BinaryIO:
        with ExitStack() as opened:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            directory = os.open(self.mount_root, directory_flags)
            opened.callback(os.close, directory)
            if self.require_mount:
                if not self.mount_root.is_absolute() or not os.path.ismount(self.mount_root):
                    raise RuntimeError("admitted volume mount is absent")
                current = os.stat(self.mount_root, follow_symlinks=False)
                bound = os.fstat(directory)
                if (current.st_dev, current.st_ino) != (bound.st_dev, bound.st_ino):
                    raise RuntimeError("admitted volume mount changed")
            for segment in segments[:-1]:
                directory = os.open(segment, directory_flags, dir_fd=directory)
                opened.callback(os.close, directory)
            descriptor = os.open(
                segments[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise PermissionError("mounted object must be a regular file")
                return os.fdopen(descriptor, "rb")
            except BaseException:
                os.close(descriptor)
                raise
