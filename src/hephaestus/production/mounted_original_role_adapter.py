"""Read-only original adapter bytes from the already mounted RunPod volume.

No S3 access key is ever passed into a GPU Pod. The existing certified
VerifiedRoleAdapters verifies the original *exact* SHA-256 and byte size
after reading each streamed object through this narrowly scoped provider.
"""
from dataclasses import dataclass
from pathlib import Path,PurePosixPath
import stat


@dataclass(frozen=True,slots=True)
class MountedReadOnlyOriginalS3Client:
    mount_root: Path
    expected_bucket: str

    def get_object(self, *, Bucket, Key):
        if Bucket!=self.expected_bucket or not isinstance(Key,str):
            raise PermissionError("mounted original adapter volume identity differs")
        normalized=PurePosixPath(Key)
        if (normalized.is_absolute() or not normalized.parts or
                any(p in (".","..") for p in normalized.parts)
                or "\\" in Key or "\x00" in Key):
            raise PermissionError("original adapter key escapes mounted read-only volume")
        root=self.mount_root.resolve(strict=True)
        item=root
        for segment in normalized.parts:
            item=item/segment
            metadata=item.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PermissionError("original adapter mount forbids symbolic links")
        final=item.resolve(strict=True)
        if root not in final.parents or not final.is_file():
            raise PermissionError("original adapter is not a regular mounted S3 artifact")
        return {"Body":final.open("rb")}
