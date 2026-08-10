"""인덱스 지문: 어댑터·데이터가 뒤바뀐 인덱스를 조용히 쓰지 않는지 고정."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from embed import check_index_meta, index_fingerprint  # noqa: E402

CFG = {"model": {"model_id": "models/base", "image_size": 224}}


def _write(tmp_path, fp):
    d = tmp_path / "index"
    d.mkdir(exist_ok=True)
    (d / "index_meta.json").write_text(json.dumps(fp), encoding="utf-8")
    return str(d)


def test_지문이_같으면_통과(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl")
    assert ok, why


def test_어댑터가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/b", "data/pairs.jsonl")
    assert not ok and "adapter" in why


def test_데이터가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/subset_100k.jsonl")
    assert not ok and "data" in why


def test_해상도가_다르면_거부(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    other = {"model": {"model_id": "models/base", "image_size": 384}}
    ok, why = check_index_meta(_write(tmp_path, fp), other, "adapters/a", "data/pairs.jsonl")
    assert not ok and "image_size" in why


def test_지문_파일이_없으면_거부(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    ok, why = check_index_meta(str(d), CFG, "adapters/a", "data/pairs.jsonl")
    assert not ok and "index_meta.json" in why
