"""메서드 레지스트리: configs/methods/<name>.yaml 한 장이 방법 하나를 규정한다.

병합은 이 모듈에서만 한다. 학습·평가·인덱스·서빙이 각자 YAML을 병합하면 네 곳의
병합 로직이 어긋나 조용히 다른 설정으로 도는 사고가 난다. resolve()가 만든
config.resolved.yaml만 이후 단계가 읽는다 — 재현성도 여기서 나온다.
"""
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS_DIR = os.path.join(ROOT, "configs", "methods")
OUT_ROOT = os.path.join(ROOT, "outputs", "methods")

DEFAULT_BASE = "configs/lora_clip.yaml"
_OVERRIDABLE = ("model", "lora", "train")     # 메서드 YAML이 덮어쓸 수 있는 최상위 블록


def deep_merge(base, over):
    """over를 base 위에 재귀 병합한 새 dict. base는 변경하지 않는다."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) \
            if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def list_methods():
    """등록된 메서드 이름 목록 (정렬)."""
    if not os.path.isdir(METHODS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(METHODS_DIR) if f.endswith(".yaml"))


def method_dir(name):
    return os.path.join(OUT_ROOT, name)


def resolve(name, base_config=None):
    """메서드 YAML을 extends 베이스와 병합한 dict. 파일 쓰기는 하지 않는다."""
    path = os.path.join(METHODS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"알 수 없는 메서드: {name} ({path})")
    spec = yaml.safe_load(open(path, encoding="utf-8")) or {}
    base_rel = base_config or spec.get("extends", DEFAULT_BASE)
    base = yaml.safe_load(open(os.path.join(ROOT, base_rel), encoding="utf-8"))

    cfg = deep_merge(base, {k: v for k, v in spec.items() if k in _OVERRIDABLE})
    cfg["method"] = {
        "name": name,
        "description": spec.get("description", ""),
        "extends": base_rel,
        "data": spec.get("data", {"builder": "shared", "pairs": "data/pairs.jsonl"}),
    }
    # 산출물 경로는 규약으로 고정 — 메서드 YAML이 지정하지 못하게 한다
    cfg["train"]["output_dir"] = os.path.relpath(method_dir(name), ROOT).replace("\\", "/")
    return cfg


def write_resolved(name, cfg=None):
    """resolved config를 outputs/methods/<name>/config.resolved.yaml로 못박고 경로 반환."""
    cfg = resolve(name) if cfg is None else cfg
    d = method_dir(name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "config.resolved.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def write_data_pointer(name, cfg):
    """방법의 데이터 위치를 산출물 디렉터리에 기록하고 실제 pairs 경로를 반환한다.

    builder가 shared면 472k 레코드를 복제하지 않고 포인터 JSON만 남긴다. 심볼릭
    링크를 쓰지 않는 이유는 Windows에서 관리자 권한이 필요하기 때문이다. 향후
    전용 데이터를 만드는 builder가 생기면 여기서 실제 파일 경로를 반환하면 된다.
    """
    data = cfg.get("method", {}).get("data", {})
    source = data.get("pairs", "data/pairs.jsonl")
    d = os.path.join(method_dir(name), "data")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pairs.jsonl.pointer.json"), "w", encoding="utf-8") as f:
        json.dump({"builder": data.get("builder", "shared"), "source": source},
                  f, ensure_ascii=False, indent=2)
    return os.path.join(ROOT, source)
