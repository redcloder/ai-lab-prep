import torch


def main() -> None:
    print("=" * 40)
    print("PyTorch environment check")
    print("=" * 40)

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is unavailable; using CPU.")

    x = torch.rand(3, 4, device=device)

    print(f"\nTensor:\n{x}")
    print(f"Tensor shape: {x.shape}")
    print(f"Tensor dtype: {x.dtype}")
    print(f"Tensor device: {x.device}")


if __name__ == "__main__":
    main()