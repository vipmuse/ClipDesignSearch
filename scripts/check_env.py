"""LoRA 학습 환경 점검: GPU/CUDA/bf16/라이브러리 버전."""
import torch


def main():
    print("=== PyTorch / CUDA ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda (build):", torch.version.cuda)
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        print("device:", torch.cuda.get_device_name(i))
        cap = torch.cuda.get_device_capability(i)
        print("compute capability:", f"sm_{cap[0]}{cap[1]}")
        props = torch.cuda.get_device_properties(i)
        print("VRAM (GB):", round(props.total_memory / 1024**3, 1))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
        # 실제 GPU 연산 스모크 테스트 (Blackwell는 커널 호환 확인 중요)
        try:
            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            (x @ x).sum().item()
            print("bf16 matmul smoke test: OK")
        except Exception as e:
            print("bf16 matmul smoke test: FAIL ->", e)
    else:
        print("!! CUDA를 못 씀 — CPU 빌드이거나 드라이버/CUDA 불일치")

    print("\n=== libs ===")
    for m in ["transformers", "peft", "faiss", "accelerate"]:
        try:
            print(m, __import__(m).__version__)
        except Exception as e:
            print(m, "MISSING ->", e)


if __name__ == "__main__":
    main()
