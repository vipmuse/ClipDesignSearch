"""도면-텍스트 쌍 데이터셋 + 도면 전처리.

data/pairs.jsonl 한 줄 예시:
  {"image": "imgs/D3012345_v1.png", "text": "무선 이어폰 케이스", "design_id": "D3012345", "locarno": "14-03"}
"""
import json
import os
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None   # 초대형 도면의 DecompressionBomb 예외 방지


def load_records(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def preprocess_drawing(img: Image.Image, size: int) -> Image.Image:
    """흑백 도면 → 여백 크롭 + 정사각 패딩 + 3채널. 색/질감 증강은 하지 않는다."""
    img = img.convert("L")               # 그레이스케일로 통일
    img = ImageOps.autocontrast(img)     # 라인 대비 강화
    img = ImageOps.invert(img)           # 여백(흰색)을 0으로 만들어 크롭 대상 명확화
    bbox = img.getbbox()                 # 도면 내용 영역
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.invert(img)           # 원래 극성으로 복귀
    # 비율 유지 축소 + 정사각 패딩. 극단적 종횡비(예: 1500x3)에서 0-크기가 나지 않게 직접 처리.
    w, h = img.size
    if w < 1 or h < 1:
        return Image.new("RGB", (size, size), (255, 255, 255))
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas.convert("RGB")         # CLIP은 3채널 입력


@dataclass
class Collator:
    """PIL 전처리 + HF CLIPProcessor 토크나이즈/정규화를 배치로 묶는다."""
    processor: object
    image_size: int
    image_root: str

    def __call__(self, batch):
        images, texts, design_ids = [], [], []
        for r in batch:
            path = os.path.join(self.image_root, r["image"])
            try:                              # 깨진/degenerate 이미지는 흰 배경으로 대체(학습 중단 방지)
                img = preprocess_drawing(Image.open(path), self.image_size)
            except Exception:
                img = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
            images.append(img)
            texts.append(r["text"])
            design_ids.append(r.get("design_id", r["image"]))

        enc = self.processor(
            text=texts, images=images,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        # 같은 design_id → 같은 정수 라벨 (이미지↔이미지 supervised contrastive용)
        uniq = {d: i for i, d in enumerate(dict.fromkeys(design_ids))}
        enc["design_label"] = torch.tensor([uniq[d] for d in design_ids], dtype=torch.long)
        return enc


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
