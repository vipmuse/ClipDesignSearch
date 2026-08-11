"""등록된 모든 메서드 YAML이 실제로 resolve되는지 확인.

YAML 오타나 extends 경로 오류를 학습 몇 시간 뒤가 아니라 지금 잡는다.
"""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import registry  # noqa: E402
from run_ablation import DEFAULT_ARMS, is_trained  # noqa: E402

METHODS = registry.list_methods()

KNOBS = ("pk_views", "locarno_aware", "mask_false_negatives", "augment", "img2img_weight")
OFF = (1, False, False, False, 0.0)


def test_메서드가_하나_이상_등록되어_있다():
    assert METHODS, "configs/methods/*.yaml 이 비어 있다"


@pytest.mark.parametrize("name", METHODS)
def test_모든_메서드가_resolve된다(name):
    cfg = registry.resolve(name)
    assert cfg["method"]["name"] == name
    for key in ("model", "lora", "train"):
        assert key in cfg, f"{name}: {key} 블록 누락"
    assert cfg["train"]["output_dir"] == f"outputs/methods/{name}"


@pytest.mark.parametrize("name", METHODS)
def test_YAML의_name_필드가_파일명과_일치한다(name):
    import yaml
    spec = yaml.safe_load(open(os.path.join(registry.METHODS_DIR, f"{name}.yaml"),
                               encoding="utf-8"))
    assert spec.get("name") == name, f"{name}.yaml 의 name 필드가 파일명과 다르다"


def _knobs(name):
    t = registry.resolve(name)["train"]
    return tuple(t[k] for k in KNOBS)


def test_baseline은_모든_개선이_꺼져있다():
    """비교의 기준점이 은근슬쩍 켜져 있으면 모든 Δ가 무의미해진다."""
    assert _knobs("baseline") == OFF, f"baseline이 기준점이 아니다: {_knobs('baseline')}"


# ── is_trained: base와 baseline이 "학습됨" 판정을 공유하는지 고정 ──
# (회귀 재발 방지: run_ablation.py의 base 분기가 이 predicate 없이 따로 조건을
#  적으면, --force 재실행이나 학습 중단 시 base와 baseline이 서로 다른
#  config.resolved.yaml을 보게 되어 비교표의 기준점이 조용히 틀어진다.)

def test_is_trained는_config만_있고_어댑터가_없으면_False(tmp_path, monkeypatch):
    """학습이 중단된 상태를 고정한다: config.resolved.yaml은 학습 시작 전에
    쓰이므로, 크래시 직후엔 config만 있고 final/adapter_config.json은 없을 수
    있다. 이때 is_trained가 True를 주면(=학습됐다고 착각하면) 다음 실행에서
    base가 이 미완성 config를 '학습된 baseline의 저장본'으로 재사용해버린다."""
    monkeypatch.setattr(registry, "OUT_ROOT", str(tmp_path / "outputs" / "methods"))
    d = registry.method_dir("baseline")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.resolved.yaml"), "w", encoding="utf-8") as f:
        f.write("train: {}\n")
    assert not os.path.exists(os.path.join(d, "final", "adapter_config.json"))

    assert is_trained("baseline", argparse.Namespace(force=False)) is False


def test_is_trained는_어댑터가_있고_force가_아니면_True(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "OUT_ROOT", str(tmp_path / "outputs" / "methods"))
    final = os.path.join(registry.method_dir("baseline"), "final")
    os.makedirs(final, exist_ok=True)
    with open(os.path.join(final, "adapter_config.json"), "w", encoding="utf-8") as f:
        f.write("{}")

    assert is_trained("baseline", argparse.Namespace(force=False)) is True
    assert is_trained("baseline", argparse.Namespace(force=True)) is False


def test_기본_arm들은_서로_다른_레시피다():
    """두 arm이 같은 설정으로 resolve되면 ablation은 Δ≈0을 찍고 '어떤 개선도 효과
    없다'는 결론이 나온다. YAML 오타(예: train→trian)나 복붙 실수가 정확히 이 모양의
    사고를 만들기 때문에, 값이 실제로 갈리는지를 여기서 못박는다."""
    seen = {}
    for name in DEFAULT_ARMS:
        k = _knobs(name)
        assert k not in seen, f"{name}과 {seen[k]}가 같은 레시피로 resolve된다: {dict(zip(KNOBS, k))}"
        seen[k] = name


def test_hobit은_baseline과_sampler만_다르다():
    """hobit이 바꾸는 축은 배치 구성 하나여야 Δ가 그 기여로 읽힌다."""
    h = registry.resolve("hobit")["train"]
    b = registry.resolve("baseline")["train"]
    five = ("pk_views", "locarno_aware", "mask_false_negatives", "augment", "img2img_weight")
    assert [h[k] for k in five] == [b[k] for k in five]
    assert h["sampler"] == "hobit"
    assert b.get("sampler", "pk") != "hobit"


def test_hobit_하이퍼파라미터가_존재한다():
    h = registry.resolve("hobit")["train"]
    assert h["hobit_pool"] >= h["batch_size"], "후보 풀이 배치보다 작으면 greedy가 무의미"
    assert h["hobit_penalty"] > 0
    assert h["hobit_refresh_every"] >= 1


def test_tic은_baseline과_손실_항만_다르다():
    """tic이 바꾸는 축은 손실 하나여야 Δ가 그 기여로 읽힌다."""
    c = registry.resolve("tic")["train"]
    b = registry.resolve("baseline")["train"]
    five = ("pk_views", "locarno_aware", "mask_false_negatives", "augment", "img2img_weight")
    assert [c[k] for k in five] == [b[k] for k in five]
    assert c["tic_weight"] > 0
    assert b.get("tic_weight", 0.0) == 0.0, "baseline에 TIC이 켜져 있으면 비교가 무의미"


@pytest.mark.parametrize("name", METHODS)
def test_tic_하이퍼파라미터가_유효_범위_안이다(name):
    """registry.resolve()는 최상위 키 오타만 막는다 — train: 안쪽은 검증하지 않는다.

    그래서 `tic_flor: 0.7`은 조용히 통과하고 학습은 기본값 floor=0.75로 돈다. 한 글자
    오타로 두 arm이 똑같이 학습되면 summary.md에는 서로 다른 floor/ceiling이 같은
    수치와 함께 찍힌다 — 표만 봐서는 절대 드러나지 않는 사고다.
    tic_weight ≤ 0도 같은 모양이다(게이트가 통째로 꺼져 이름만 tic인 baseline).
    """
    t = registry.resolve(name)["train"]
    if "tic_weight" not in t:
        pytest.skip(f"{name}: TIC 미사용")
    assert isinstance(t["tic_weight"], (int, float)) and not isinstance(t["tic_weight"], bool), \
        f"{name}: tic_weight가 숫자가 아니다({t['tic_weight']!r}) — YAML 따옴표 확인"
    assert t["tic_weight"] > 0, f"{name}: tic_weight={t['tic_weight']} → 게이트가 꺼진 채 학습된다"
    # tic_floor/tic_ceiling은 명시 필수. 없으면 오타로 흘렸다는 뜻이고,
    # 학습은 조용히 기본값으로 돈다.
    assert "tic_floor" in t, "tic_floor 누락 — 키 오타면 기본값으로 조용히 넘어간다"
    assert "tic_ceiling" in t, "tic_ceiling 누락"
    assert 0 < t["tic_floor"] < t["tic_ceiling"] < 1, \
        f"floor < ceiling < 1 이어야 한다: floor={t['tic_floor']} ceiling={t['tic_ceiling']}"


@pytest.mark.parametrize("name", METHODS)
def test_알_수_없는_tic_키가_없다(name):
    """`tic_flor: 0.7` 같은 오타를 이름 그대로 잡는다. resolve()가 최상위 키에 하는
    일을 tic_* 접두어에 한정해 train: 안쪽에서도 한다."""
    t = registry.resolve(name)["train"]
    unknown = sorted(k for k in t if k.startswith("tic_")
                      and k not in ("tic_weight", "tic_floor", "tic_ceiling"))
    assert not unknown, f"{name}: 알 수 없는 TIC 키 {unknown} (허용: tic_weight, tic_floor, tic_ceiling)"


def test_tic과_hobit은_서로_다른_축을_바꾼다():
    """두 방법이 같은 축을 건드리면 기여도가 섞인다."""
    tic = registry.resolve("tic")["train"]
    hobit = registry.resolve("hobit")["train"]
    assert tic.get("sampler", "pk") != "hobit", "tic이 배치 구성까지 바꾸고 있다"
    assert hobit.get("tic_weight", 0.0) == 0.0, "hobit이 손실까지 바꾸고 있다"


def test_loracap은_baseline과_lora_블록만_다르다():
    """loracap이 바꾸는 축은 파라미터 용량 하나여야 Δ가 그 기여로 읽힌다."""
    import yaml
    c = registry.resolve("loracap")
    b = registry.resolve("baseline")
    five = ("pk_views", "locarno_aware", "mask_false_negatives", "augment", "img2img_weight")
    assert [c["train"][k] for k in five] == [b["train"][k] for k in five]
    # resolved끼리 비교하면 둘 다 model 블록이 없어 항상 참이라 아무것도 검증하지 못한다.
    # YAML 원문에 model 블록이 없다는 것을 봐야 한다 - 해상도 축은 hires378의 몫이고,
    # 여기에 model이 생기면 두 방법이 같은 축을 건드리게 된다.
    spec = yaml.safe_load(open(os.path.join(registry.METHODS_DIR, "loracap.yaml"),
                               encoding="utf-8"))
    assert "model" not in spec, "loracap.yaml에 model 오버라이드가 생겼다 - 해상도 축 침범"


def test_loracap의_lora_블록이_베이스보다_크다():
    """세 가지가 모두 켜져야 '용량 확장'이다 - 하나라도 빠지면 다른 arm과 구분되지 않는다."""
    c = registry.resolve("loracap")["lora"]
    b = registry.resolve("baseline")["lora"]
    assert c["r"] > b["r"] and c["alpha"] > b["alpha"]
    assert set(c["target_modules"]) > set(b["target_modules"]), "fc1/fc2가 추가되지 않았다"
    assert c["train_projections"] is True and b["train_projections"] is False


def test_loracap은_손실과_배치_구성_축을_건드리지_않는다():
    """tic(손실 축)·hobit(배치 구성 축)과 겹치면 기여도가 섞인다."""
    c = registry.resolve("loracap")["train"]
    assert c.get("tic_weight", 0.0) == 0.0
    assert c.get("sampler", "random") != "hobit"
