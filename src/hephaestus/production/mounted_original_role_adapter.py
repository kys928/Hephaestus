"""Compatibility import for the original certified regional worker."""

from hephaestus.storage.mounted_read import (
    MountedReadOnlyObjectSource as MountedReadOnlyOriginalS3Client,
)

__all__ = ["MountedReadOnlyOriginalS3Client"]
