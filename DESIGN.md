# CLIP + LoRA 디자인/특허 특화 튜닝 설계

CLIP 백본을 **프리징(freeze)** 하고, **LoRA 어댑터만 학습**하여
디자인권(의장)·특허 도면 검색에 특화시키는 파이프라인 설계.

> **정확도 개선 분석·구현 내역: [ACCURACY.md](ACCURACY.md)**
> (PK 샘플러, false-negative 마스킹, design 단위 평가 분할, 전체 갤러리 mAP 평가,
> 검색 단계 αQE/TTA/멀티뷰 집계 — 2026-08 반영)

---

## 0. 베이스 모델 (확정: MetaCLIP 2)

**MetaCLIP 2 (worldwide)** 로 확정. 300+ 언어로 학습된 최신 CLIP 계열로 다국어(한국어 포함)
명세서 텍스트 검색에 유리하고, XM3600 등 다국어 벤치에서 mSigLIP·SigLIP-2를 상회.

- 기본 체크포인트: `facebook/metaclip-2-worldwide-huge-quickgelu`
- 경량 PoC 대안: `facebook/metaclip-2-worldwide-s16` (ViT-S/16), `-s16-384`(고해상도)
- **로딩 주의**: `CLIPModel`이 아니라 `AutoModel`/`AutoProcessor`(내부 `MetaClip2Model`)로 로드.
  단, 서브모듈 구조(`vision_model`/`text_model`, `self_attn.{q,k,v,out}_proj`, `logit_scale`,
  출력 `logits_per_text`/`image_embeds`)는 CLIP과 동일 계열이라 LoRA·학습 코드는 그대로 호환.
- 필요 시 `transformers>=4.52`(MetaCLIP 2 지원 버전) 확인.

| 후보 | HF ID | 비고 |
|------|-------|------|
| **MetaCLIP 2 (확정)** | `facebook/metaclip-2-worldwide-huge-quickgelu` | 다국어 최신, 본 설계 기본 |
| OpenCLIP ViT-L/14 | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 범용, `CLIPModel`로 로드 |
| Jina-CLIP v2 | `jinaai/jina-clip-v2` | 다국어·고해상도 |
| SigLIP 2 | `google/siglip2-*` | sigmoid loss, 리트리벌 강함 |

> 도면(라인 드로잉) 도메인 갭이 크므로, 베이스 성능보다 **LoRA 튜닝 데이터 품질**이 결과를 좌우합니다.
> 다른 모델로 교체 시 `configs/lora_clip.yaml`의 `model_id`만 바꾸면 되며,
> `CLIPModel` 전용 모델은 `src/model.py`·`src/embed.py`의 `AutoModel`을 `CLIPModel`로 되돌리면 됩니다.

## 0-1. 검색 시나리오 (확정: 양방향)

- **텍스트→도면**: 물품 명칭/설명으로 유사 디자인 검색 (CLIP 이미지↔텍스트 loss)
- **도면→도면**: 도면 이미지 쿼리로 유사 도면 검색 (같은 `design_id` supervised contrastive)
- → `img2img_weight: 0.5`로 두 신호를 함께 학습. 평가도 T→I / I→T 양방향 Recall 로깅.

---

## 1. 도메인 특성과 설계 제약

디자인/특허 도면은 자연 이미지와 다릅니다:

- **흑백 라인 드로잉**, 음영/질감 없음 → 색·텍스처 기반 특징이 무의미
- **다중 뷰**(정면·측면·평면·사시도)가 한 디자인을 구성
- **로카르노 분류(Locarno)** / IPC·CPC 등 구조화된 라벨 존재
- 텍스트는 **물품 명칭 + 디자인 설명 + 청구항** (한국어/영어 혼재)
- 클래스 내 미세한 형태 차이가 핵심 (fine-grained)

→ 설계 함의:
1. 전처리에서 **흑백 도면에 맞는 정규화 + 뷰 처리** 필요
2. 학습 신호는 `(도면, 텍스트)` 쌍 + `(도면, 도면)` 유사쌍(같은 출원/클래스) 둘 다 활용
3. 평가는 **Recall@K / mAP** 기반 리트리벌 메트릭

---

## 2. 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │         CLIP (Frozen 🔒)              │
  도면 이미지 ──────►│  Vision Encoder ─┐                   │
                    │                  ├─► logit_scale ──► 유사도
  텍스트/명칭 ──────►│  Text Encoder  ──┘                   │
                    └───────▲──────────────────▲───────────┘
                            │                  │
                     LoRA A/B (🔥학습)   LoRA A/B (🔥학습)
                     q,k,v,out_proj      q,k,v,out_proj
                     (+ mlp, projection 선택)
```

- **프리징**: `requires_grad=False`로 전 파라미터 동결
- **LoRA**: attention 투영층(`q_proj,k_proj,v_proj,out_proj`)에 저랭크 어댑터 주입
  - rank `r=16`, `alpha=32`, dropout `0.05` 기본
  - vision·text 양쪽 인코더 모두 대상 (도메인 갭이 이미지·텍스트 모두 존재)
- **학습 파라미터**: LoRA 어댑터 + `logit_scale`(온도) 만 → 전체의 ~1~3%
- 옵션: `visual_projection`, `text_projection`도 LoRA 대상에 포함하면 임베딩 공간 정렬 개선

### 학습 대상 요약
| 구성요소 | 상태 |
|---|---|
| Vision/Text Transformer 블록 | 🔒 Frozen |
| LoRA A/B (attention) | 🔥 Trainable |
| logit_scale | 🔥 Trainable (clamp ≤ 100) |
| LayerNorm | 🔒 (선택적으로 unfreeze 가능) |

---

## 3. 데이터 파이프라인

### 입력 포맷 (`data/pairs.jsonl`)
```json
{"image": "imgs/D3012345_v1.png", "text": "무선 이어폰 케이스, 로카르노 14-03", "design_id": "D3012345", "locarno": "14-03"}
{"image": "imgs/D3012345_v2.png", "text": "무선 이어폰 케이스 측면도",       "design_id": "D3012345", "locarno": "14-03"}
```

- `design_id`: 같은 출원의 여러 뷰 → **positive 쌍** 구성에 사용
- `locarno`: 하드 네거티브/클래스 균형 샘플링에 사용

### 전처리
- 도면 흑백 → 3채널 복제 후 CLIP 정규화
- 여백 크롭 + 정사각 패딩 (도면 비율 보존, 왜곡 방지)
- 증강: 가벼운 회전(±10°), 소폭 스케일, 랜덤 라인 두께 — **색 증강 금지**

### 학습 신호 (2가지 loss 조합)
1. **이미지↔텍스트 대조학습** (CLIP InfoNCE, `return_loss=True`)
2. **이미지↔이미지 대조학습** (같은 `design_id` = positive) — 뷰 불변 표현 학습
   - 배치 내 `design_id` 그룹으로 supervised contrastive 적용

---

## 4. 학습 전략

| 항목 | 값 | 이유 |
|---|---|---|
| Optimizer | AdamW | LoRA 표준 |
| LR (LoRA) | 1e-4 ~ 5e-4 | 어댑터는 높은 LR 허용 |
| LR (logit_scale) | 1e-5 | 온도는 천천히 |
| Scheduler | cosine + warmup 5% | 안정화 |
| Batch | 가능한 크게(대조학습은 배치=네거티브 수) | grad accumulation 활용 |
| Precision | bf16 | 메모리·속도 |
| Epochs | 5~15 | 소규모 데이터 과적합 주의, early stop |
| Freeze | 백본 전체 | 목표 |

- **평가**: 홀드아웃 도면 세트로 Text→Image / Image→Image **Recall@1/5/10, mAP**
- 베이스라인(튜닝 전 CLIP) 대비 개선폭을 반드시 로깅

---

## 5. 추론/서빙 (검색)

1. 학습 후 LoRA 어댑터 저장 (`adapter_model.safetensors`, 수 MB)
2. 전체 도면 DB를 인코딩 → 벡터 인덱스(FAISS) 구축
3. 쿼리(텍스트 or 도면) 인코딩 → top-K 근접 검색
4. LoRA는 `merge_and_unload()`로 백본에 병합 가능(추론 지연 0)

---

## 6. 파일 구조
```
ClipDesignSearch/
├── DESIGN.md                 # 본 문서
├── requirements.txt
├── configs/lora_clip.yaml    # 모델/LoRA/학습 하이퍼파라미터
├── data/pairs.jsonl          # 학습 데이터(도면-텍스트 쌍)
└── src/
    ├── dataset.py            # JSONL 로더 + 도면 전처리
    ├── model.py              # CLIP 로드 + 프리징 + LoRA 주입
    ├── train.py              # 대조학습 루프 + 평가
    └── embed.py              # DB 인코딩 + FAISS 검색
```

## 7. 마일스톤
1. **PoC**: 소량(1~2천 쌍)으로 파이프라인 검증, 베이스라인 대비 Recall 확인
2. **스케일업**: 데이터 확장 + 이미지↔이미지 loss 추가, 하이퍼파라미터 튜닝
3. **서빙**: FAISS 인덱스 + 어댑터 병합, API 래핑
