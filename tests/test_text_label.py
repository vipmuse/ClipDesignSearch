"""text_label 재도입: 원문 기준이며 positive 판정에는 쓰이지 않는다.

Phase 0에서 이 라벨을 제거한 이유는 _pos_mask가 이것을 positive 근거로 써서
마스크가 거의 전부 True가 됐기 때문이다. 지금 용도는 정반대 — TIC이 밀어낼
쌍에서 '제목이 같은 쌍'을 빼기 위한 필터다.
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import Collator  # noqa: E402


class _FakeProcessor:
    """텍스트 개수만 맞춰 최소 텐서를 돌려주는 가짜. 토크나이저를 로드하지 않는다."""

    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        n = len(text)
        return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long),
                "pixel_values": torch.zeros(n, 3, 8, 8)}


def _records(texts, designs, viewpoints=None):
    return [{"image": f"{i}.png", "text": t, "design_id": d,
             "viewpoint": (viewpoints[i] if viewpoints else "")}
            for i, (t, d) in enumerate(zip(texts, designs))]


def _collate(records, image_root, augment=False):
    return Collator(_FakeProcessor(), 8, image_root, augment=augment)(records)


def test_같은_원문_제목은_같은_라벨(tmp_path):
    for i in range(3):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    enc = _collate(_records(["Shoe", "Shoe", "Bottle"], ["A", "B", "C"]), str(tmp_path))
    tl = enc["text_label"].tolist()
    assert tl[0] == tl[1] and tl[0] != tl[2]


def test_증강이_켜져도_라벨은_원문_기준이다(tmp_path):
    """옛 구현은 viewpoint가 붙은 텍스트로 라벨을 만들어 같은 제목이 배치마다
    다른 라벨을 받았다. 증강 확률과 무관하게 라벨이 갈리지 않아야 한다."""
    for i in range(2):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    recs = _records(["Shoe", "Shoe"], ["A", "B"], viewpoints=["front view", "side view"])
    labels = set()
    for _ in range(30):                       # 증강은 확률적 → 여러 번 돌려 확인
        enc = _collate(recs, str(tmp_path), augment=True)
        tl = enc["text_label"].tolist()
        assert tl[0] == tl[1], "증강 때문에 같은 제목이 다른 라벨을 받았다"
        labels.add(tuple(tl))
    assert labels == {(0, 0)}


def test_design_label도_함께_나온다(tmp_path):
    for i in range(2):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / f"{i}.png")
    enc = _collate(_records(["Shoe", "Bottle"], ["A", "A"]), str(tmp_path))
    assert enc["design_label"].tolist() == [0, 0]


def test_pos_mask는_text_label을_받지_않는다():
    """Phase 0의 회귀 방지 — 제목 동일성이 positive 근거로 돌아오면 안 된다."""
    import inspect

    from train import _pos_mask
    assert list(inspect.signature(_pos_mask).parameters) == ["design_label"]
