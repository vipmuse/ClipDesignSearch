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
