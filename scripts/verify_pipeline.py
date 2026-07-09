"""엔드투엔드 검증: 로컬 MetaCLIP 2 로드 → 프리징 → LoRA 주입 → GPU forward.

  python scripts/verify_pipeline.py
"""
import os
import sys

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model  # noqa: E402


def main():
    cfg = yaml.safe_load(open("configs/lora_clip.yaml", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">> 모델 로드 + LoRA 주입")
    model, proc = build_model(cfg)          # print_trainable_parameters() 자동 출력
    model.to(device)

    # 프리징 검증: 학습 파라미터 비율
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f">> trainable {trainable:,} / total {total:,} = {100*trainable/total:.2f}%")

    # 더미 입력 (흰 배경 도면 흉내 2장 + 텍스트 2개)
    imgs = [Image.new("RGB", (cfg["model"]["image_size"],) * 2, "white") for _ in range(2)]
    texts = ["무선 이어폰 케이스 정면도", "wireless earbud case front view"]
    enc = proc(text=texts, images=imgs, return_tensors="pt",
               padding=True, truncation=True, max_length=77).to(device)

    print(">> forward (bf16 autocast, return_loss)")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"], return_loss=True)

    # 학습 코드가 쓰는 출력 속성 존재/형상 확인
    print("  loss:", float(out.loss.detach()))
    print("  logits_per_text:", tuple(out.logits_per_text.shape))
    print("  logits_per_image:", tuple(out.logits_per_image.shape))
    print("  image_embeds:", tuple(out.image_embeds.shape))
    print("  text_embeds:", tuple(out.text_embeds.shape))

    # backward가 LoRA 파라미터로 흐르는지 확인
    out.loss.backward()
    g = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    print(f">> grad 흐른 학습 파라미터 수: {len(g)} (예: {g[0] if g else 'NONE'})")

    print(f">> peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("\nALL OK - 학습 파이프라인 준비 완료")


if __name__ == "__main__":
    main()
