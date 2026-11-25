"""Exact permitted provider SKUs and CUDA names for the bounded BF16 pilot."""
import re

GPU_NAMES = {
    "NVIDIA A40": ("NVIDIA A40",),
    "NVIDIA RTX A6000": ("NVIDIA RTX A6000",),
    "NVIDIA RTX 6000 Ada Generation": ("NVIDIA RTX 6000 Ada Generation",),
    "NVIDIA L40": ("NVIDIA L40",),
    "NVIDIA L40S": ("NVIDIA L40S",),
    "NVIDIA A100 80GB PCIe": ("NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-80GB"),
    "NVIDIA A100-SXM4-80GB": ("NVIDIA A100-SXM4-80GB",),
}
MIN_BYTES = 44 * 1024**3  # Allow ECC/driver reservation on nominal 48GB devices.

def valid_region(region):
    return isinstance(region,str) and re.fullmatch(r"[A-Z]{2}-[A-Z0-9]+-[0-9]+",region) is not None

def check_cuda(torch, sku):
    if (sku not in GPU_NAMES or not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0) not in GPU_NAMES[sku]
            or not torch.cuda.is_bf16_supported()
            or torch.cuda.get_device_properties(0).total_memory<MIN_BYTES
            or torch.cuda.get_device_capability(0) not in {(8,0),(8,6),(8,9)}):
        raise PermissionError("selected GPU failed exact CUDA qualification")
    return {"gpu_sku":sku,"gpu_name":torch.cuda.get_device_name(0),
            "vram_bytes":torch.cuda.get_device_properties(0).total_memory,
            "compute_capability":list(torch.cuda.get_device_capability(0))}
