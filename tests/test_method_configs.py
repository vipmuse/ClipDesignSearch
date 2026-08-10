"""등록된 모든 메서드 YAML이 실제로 resolve되는지 확인.

YAML 오타나 extends 경로 오류를 학습 몇 시간 뒤가 아니라 지금 잡는다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import registry  # noqa: E402
from run_ablation import DEFAULT_ARMS  # noqa: E402

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


def test_기본_arm들은_서로_다른_레시피다():
    """두 arm이 같은 설정으로 resolve되면 ablation은 Δ≈0을 찍고 '어떤 개선도 효과
    없다'는 결론이 나온다. YAML 오타(예: train→trian)나 복붙 실수가 정확히 이 모양의
    사고를 만들기 때문에, 값이 실제로 갈리는지를 여기서 못박는다."""
    seen = {}
    for name in DEFAULT_ARMS:
        k = _knobs(name)
        assert k not in seen, f"{name}과 {seen[k]}가 같은 레시피로 resolve된다: {dict(zip(KNOBS, k))}"
        seen[k] = name
