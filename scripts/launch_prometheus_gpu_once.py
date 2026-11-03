"""One-time operator CLI using Hephaestus's tested RunPod admission service."""
from hephaestus.development.prometheus_gpu_provider import main

if __name__ == "__main__":
    raise SystemExit(main())
