"""도면-텍스트 쌍 데이터셋 + 도면 전처리 + PK 배치 샘플러.

data/pairs.jsonl 한 줄 예시:
  {"image": "imgs/D3012345_v1.png", "text": "무선 이어폰 케이스", "design_id": "D3012345", "locarno": "14-03"}
"""
import json
import math
import os
import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageFilter, ImageOps
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


def split_by_design(records, eval_ratio, seed):
    """design_id 단위 train/eval 분할.

    레코드 단위로 나누면 같은 디자인의 뷰들이 양쪽에 갈려 eval이 과대평가됨(누수).
    반환: (train_records, eval_records)
    """
    groups = {}
    for r in records:
        groups.setdefault(r.get("design_id", r["image"]), []).append(r)
    ids = sorted(groups)                       # 결정적 순서 위에서 시드 셔플
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_eval = max(1, int(len(ids) * eval_ratio))
    eval_ids = set(ids[:n_eval])
    train, evals = [], []
    for d, rs in groups.items():
        (evals if d in eval_ids else train).extend(rs)
    return train, evals


class PKBatchSampler:
    """배치 = 디자인 P개 × 뷰 ≤K장 (re-ID의 PK 샘플링).

    랜덤 배치에서는 같은 design_id 뷰 2장이 한 배치에 들어올 확률이 사실상 0이라
    img2img supervised contrastive가 발화하지 않음 → 뷰들을 배치에 강제로 모은다.
    locarno_aware=True면 같은 로카르노 클래스(앞 2자리) 디자인끼리 배치를 구성해
    하드 네거티브를 제공한다.

    DataLoader(batch_sampler=...)로 사용. 에폭마다 다른 셔플(seed+epoch).
    """

    def __init__(self, records, batch_size, views_per_design=4,
                 locarno_aware=True, seed=42, drop_last=True):
        self.batch_size = batch_size
        self.k = max(1, views_per_design)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.by_design = {}
        for i, r in enumerate(records):
            self.by_design.setdefault(r.get("design_id", r["image"]), []).append(i)
        self.locarno_aware = locarno_aware
        if locarno_aware:
            self.by_class = {}
            for d, idxs in self.by_design.items():
                cls = (records[idxs[0]].get("locarno") or "")[:2] or "??"
                self.by_class.setdefault(cls, []).append(d)
        self.n = len(records)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        # 디자인별 뷰를 K개 단위 청크로 → 에폭에 전체 데이터 사용
        chunk_map = {}
        for d, idxs in self.by_design.items():
            idxs = idxs[:]
            rng.shuffle(idxs)
            chunk_map[d] = [idxs[i:i + self.k] for i in range(0, len(idxs), self.k)]

        if self.locarno_aware:                 # 클래스끼리 인접 배치 → 하드 네거티브
            order = []
            classes = list(self.by_class)
            rng.shuffle(classes)
            for c in classes:
                designs = self.by_class[c][:]
                rng.shuffle(designs)
                for d in designs:
                    order.extend(chunk_map[d])
        else:
            order = [ch for chunks in chunk_map.values() for ch in chunks]
            rng.shuffle(order)

        batch = []
        for ch in order:
            batch.extend(ch)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self.n // self.batch_size if self.drop_last \
            else math.ceil(self.n / self.batch_size)


def preprocess_drawing(img: Image.Image, size: int, augment: bool = False) -> Image.Image:
    """흑백 도면 → 여백 크롭 + 정사각 패딩 + 3채널. 색/질감 증강은 하지 않는다.

    augment=True(학습 전용): 라인 두께 변화 + 소회전(±7°) + 콘텐츠 스케일(0.85~1.0)
    + 랜덤 배치 오프셋. eval/추론은 결정적 전처리 유지.
    """
    img = img.convert("L")               # 그레이스케일로 통일
    img = ImageOps.autocontrast(img)     # 라인 대비 강화
    img = ImageOps.invert(img)           # 여백(흰색)을 0으로 만들어 크롭 대상 명확화
    bbox = img.getbbox()                 # 도면 내용 영역
    if bbox:
        img = img.crop(bbox)
    img = ImageOps.invert(img)           # 원래 극성으로 복귀

    if augment:
        r = random.random()              # 라인 두께: L모드에서 Min=선 굵게, Max=선 얇게
        if r < 0.15:
            img = img.filter(ImageFilter.MinFilter(3))
        elif r < 0.30:
            img = img.filter(ImageFilter.MaxFilter(3))
        angle = random.uniform(-7, 7)    # 소회전, 흰 배경 채움
        img = img.rotate(angle, expand=True, fillcolor=255, resample=Image.BILINEAR)

    # 비율 유지 축소 + 정사각 패딩. 극단적 종횡비(예: 1500x3)에서 0-크기가 나지 않게 직접 처리.
    w, h = img.size
    if w < 1 or h < 1:
        return Image.new("RGB", (size, size), (255, 255, 255))
    scale = min(size / w, size / h)
    if augment:
        scale *= random.uniform(0.85, 1.0)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh))
    canvas = Image.new("L", (size, size), 255)
    if augment:                          # 중앙 고정 대신 랜덤 오프셋 (위치 불변성)
        ox, oy = random.randint(0, size - nw), random.randint(0, size - nh)
    else:
        ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (ox, oy))
    return canvas.convert("RGB")         # CLIP은 3채널 입력


@dataclass
class Collator:
    """PIL 전처리 + HF CLIPProcessor 토크나이즈/정규화를 배치로 묶는다.

    augment=True: 이미지 증강 + 확률적(30%) viewpoint 텍스트 부가 (학습 전용).
    design_label: 같은 design_id를 positive로 묶는 멀티-positive loss
    (masked InfoNCE, supcon)용 정수 라벨.
    """
    processor: object
    image_size: int
    image_root: str
    augment: bool = False

    def __call__(self, batch):
        images, texts, design_ids = [], [], []
        for r in batch:
            path = os.path.join(self.image_root, r["image"])
            try:                              # 깨진/degenerate 이미지는 흰 배경으로 대체(학습 중단 방지)
                img = preprocess_drawing(Image.open(path), self.image_size, self.augment)
            except Exception:
                img = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
            images.append(img)
            text = r["text"]
            if self.augment and r.get("viewpoint") and random.random() < 0.3:
                text = f"{text}, {r['viewpoint']}"     # 텍스트 다양화 (④)
            texts.append(text)
            design_ids.append(r.get("design_id", r["image"]))

        enc = self.processor(
            text=texts, images=images,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        # 같은 design_id → 같은 정수 라벨 (이미지↔이미지 supervised contrastive용)
        uniq_d = {d: i for i, d in enumerate(dict.fromkeys(design_ids))}
        enc["design_label"] = torch.tensor([uniq_d[d] for d in design_ids], dtype=torch.long)
        return enc


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
