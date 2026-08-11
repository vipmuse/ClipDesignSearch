"""MetaCLIP 2 로드 → 전체 프리징 → LoRA 어댑터 주입.

MetaCLIP 2는 CLIPModel이 아니라 MetaClip2Model(=AutoModel) 클래스로 로드한다.
서브모듈 구조(vision_model/text_model, self_attn.{q,k,v,out}_proj, logit_scale)는
CLIP과 동일 계열이라 LoRA target_modules와 학습 코드가 그대로 호환된다.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor


def _resolve_target_modules(cfg_lora):
    """vision/text 인코더 선택 여부 + projection 학습 여부를 target_modules로 변환.

    PEFT는 모듈 이름 '접미사'로 매칭하므로 q_proj 등만 주면 vision·text 양쪽에 모두 붙는다.
    한쪽만 원하면 전체 경로 접두사를 붙여 필터링한다.
    """
    base = list(cfg_lora["target_modules"])
    both = cfg_lora.get("apply_to_vision", True) and cfg_lora.get("apply_to_text", True)

    if both:
        targets = base
    else:
        targets = []
        if cfg_lora.get("apply_to_vision", True):
            targets += [f"vision_model.encoder.layers.{n}" for n in base]  # 접두사 매칭
        if cfg_lora.get("apply_to_text", True):
            targets += [f"text_model.encoder.layers.{n}" for n in base]

    if cfg_lora.get("train_projections", False):
        targets += ["visual_projection", "text_projection"]
    return targets


def build_model(cfg):
    model_id = cfg["model"]["model_id"]
    model = AutoModel.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id)

    # 1) 전체 동결 (백본 프리징)
    for p in model.parameters():
        p.requires_grad = False

    # 1-1) 그래디언트 체크포인팅(옵션): 활성값을 저장하지 않고 backward에서 재계산한다.
    #      수학적으로 동일한 연산이라 배치·네거티브·손실이 그대로고 VRAM만 급감한다
    #      (배치를 줄여 VRAM을 맞추면 네거티브 수 축까지 같이 바뀌어 비교가 섞인다).
    #      PEFT 래핑 전 베이스 모델에 걸어야 어댑터가 들어갈 하위 인코더까지 적용된다.
    #      use_reentrant=False가 필수다: reentrant 구현은 체크포인트 구간의 입력이
    #      requires_grad여야 해서, 백본을 전부 동결한 이 설정에선 "element 0 of tensors
    #      does not require grad"로 죽는다(그 경우의 정석 대응이 enable_input_require_grads).
    #      비-reentrant는 그 제약이 없어 실측으로도 불필요했다: 체크포인팅 on/off와
    #      enable_input_require_grads 유무 세 조합이 loss 1.39835238, grad norm 0.96503025,
    #      grad를 받은 파라미터 339개(vision 192/text 144)까지 완전히 동일했다.
    if cfg.get("train", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # 2) LoRA 주입 (어댑터만 학습 가능 상태로 생성됨)
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=_resolve_target_modules(cfg["lora"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # 3) 온도(logit_scale)는 대조학습에서 함께 학습
    model.base_model.model.logit_scale.requires_grad = True

    model.print_trainable_parameters()
    return model, processor
