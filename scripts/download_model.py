"""베이스 모델(MetaCLIP 2)을 프로젝트 models/ 폴더에 다운로드.

HF 캐시(기본 C:) 대신 프로젝트 로컬(D:)로 받아 소스에 포함시킨다.

  python scripts/download_model.py                       # config의 model_id 사용
  python scripts/download_model.py --model-id facebook/metaclip-2-worldwide-s16
"""
import argparse
import os

import yaml
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_clip.yaml")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model_id = args.model_id or cfg["model"]["model_id"]
    local_dir = os.path.join(args.out, model_id.split("/")[-1])

    print(f"downloading {model_id} -> {local_dir}")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        # 안전텐서 우선, 중복 pytorch_model.bin 제외로 용량 절약
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*merges*", "*vocab*"],
    )
    print(f"done: {path}")
    print("config의 model_id를 로컬 경로로 바꾸려면:")
    print(f'  model_id: "{local_dir}"')


if __name__ == "__main__":
    main()
