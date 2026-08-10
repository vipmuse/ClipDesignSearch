"""인덱스 지문: 어댑터·데이터가 뒤바뀐 인덱스를 조용히 쓰지 않는지 고정."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import embed  # noqa: E402
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


def test_method_이름만_다르면_허용(tmp_path):
    """method 이름은 참고용 — 판정에 쓰지 않는다(브리프 명시 동작). 나머지가 전부
    같으면 method가 달라도 통과해야 한다."""
    cfg_a = {**CFG, "method": {"name": "hobit"}}
    cfg_b = {**CFG, "method": {"name": "tic"}}
    fp = index_fingerprint(cfg_a, "adapters/a", "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), cfg_b, "adapters/a", "data/pairs.jsonl")
    assert ok and why == ""


def test_모델이_다르면_거부(tmp_path):
    """model_id를 지문에서 빼도 나머지 테스트가 전부 통과해버려 구멍이었다."""
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    other = {"model": {"model_id": "models/other", "image_size": 224}}
    ok, why = check_index_meta(_write(tmp_path, fp), other, "adapters/a", "data/pairs.jsonl")
    assert not ok and "model_id" in why


# ── limit: 데모 인덱스를 전체 갤러리로 착각하지 않기 ──

def test_limit이_다르면_거부(tmp_path):
    """--quick(--limit 2000)이 만든 2,000장 인덱스는 전체 인덱스와 model_id·adapter·
    data가 전부 같다. limit이 지문에 없으면 서버가 데모를 전체로 믿고 서빙한다."""
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl", limit=2000)
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl")
    assert not ok and "limit" in why


def test_limit이_같으면_통과(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl", limit=2000)
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl",
                               limit=2000)
    assert ok, why


def test_limit이_지문에_기록된다(tmp_path):
    assert index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl", limit=2000)["limit"] == 2000
    assert index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")["limit"] == 0


# ── 경로 정규화: 저장소를 옮겨도 지문이 살아 있어야 한다 ──

def test_저장소_안의_절대경로는_상대경로로_정규화된다():
    """머신 절대경로가 박히면 저장소를 복제·이동한 순간 전부 불일치가 된다."""
    abs_adapter = os.path.join(embed.ROOT, "outputs", "methods", "baseline", "final")
    fp = index_fingerprint(CFG, abs_adapter, os.path.join(embed.ROOT, "data", "pairs.jsonl"))
    assert fp["adapter"] == "outputs/methods/baseline/final"
    assert fp["data"] == "data/pairs.jsonl"


def test_절대경로로_만든_인덱스를_상대경로로_조회해도_통과(tmp_path):
    """빌드는 절대경로로, 서버는 resolved config의 상대경로로 부른다 — 만나야 한다."""
    abs_adapter = os.path.join(embed.ROOT, "outputs", "methods", "baseline", "final")
    fp = index_fingerprint(CFG, abs_adapter, "data/pairs.jsonl")
    ok, why = check_index_meta(_write(tmp_path, fp), CFG,
                               "outputs/methods/baseline/final", "data/pairs.jsonl")
    assert ok, why


def test_저장소_밖_절대경로는_그대로_둔다():
    outside = os.path.join(os.path.dirname(embed.ROOT), "다른저장소", "final")
    fp = index_fingerprint(CFG, outside, "data/pairs.jsonl")
    assert fp["adapter"] == outside.replace(os.sep, "/")


# ── n_vectors: 지문이 아니라 결과라 별도 대조 ──

def test_실제_벡터수가_다르면_거부(tmp_path):
    """n_vectors는 빌드 결과(열리지 않는 도면 수를 요청 측이 알 수 없다)라 want에
    넣을 수 없다. FAISS 인덱스를 연 호출자가 index.ntotal을 넘기면 그때 대조한다."""
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    fp["n_vectors"] = 472615
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl",
                               n_vectors=2000)
    assert not ok and "n_vectors" in why


def test_실제_벡터수가_같으면_통과(tmp_path):
    fp = index_fingerprint(CFG, "adapters/a", "data/pairs.jsonl")
    fp["n_vectors"] = 2000
    ok, why = check_index_meta(_write(tmp_path, fp), CFG, "adapters/a", "data/pairs.jsonl",
                               n_vectors=2000)
    assert ok, why
