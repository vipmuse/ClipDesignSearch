# 정확도 개선 분석 & 구현 내역

코드 전체(학습·데이터·평가·서빙) 분석 결과와, 그에 따른 구현 사항 정리.
우선순위는 **효과 대비 비용** 순. ①~⑬ 번호는 본 문서 전체에서 공유.

---

## 1. 진단 (2026-08 분석)

### ① img2img supcon loss가 사실상 미발화 — **치명적** [구현됨]
`train.py`가 레코드를 랜덤 셔플해 배치를 만들었음. 디자인 수만 건 × 뷰 5~10장
규모에서 배치 32를 랜덤으로 뽑으면 **같은 `design_id` 뷰 2장이 한 배치에 들어올
확률이 사실상 0** → `supcon_loss`는 positive 없음(`valid.sum()==0`)으로 매번 0을
반환, `img2img_weight: 0.5`가 무의미했음. 도면→도면 검색용 학습 신호가 실제로는
거의 안 들어가고 있었던 것.

**해결: PK 배치 샘플러** (re-ID 표준 기법) — 배치를 "디자인 P개 × 뷰 ≤K장"으로
구성. 같은 로카르노 클래스 디자인끼리 배치를 구성하면(**locarno_aware**) 하드
네거티브까지 확보.

### ② 평가 데이터 누수 — 기존 Recall은 부풀려진 수치 [구현됨]
train/eval을 **레코드 단위**로 분할 → 같은 디자인의 뷰들이 양쪽에 갈림. 거의
동일한 도면+동일 텍스트가 학습에 들어가므로 eval이 과대평가됨.
또한 배치 내(후보 32개) Recall은 R@10 랜덤 베이스라인이 31%라 변별력이 낮음.

**해결**: (a) `design_id` 단위 분할(`split_by_design`), (b) 전체 홀드아웃 갤러리
대상 오프라인 평가 스크립트 `src/eval_retrieval.py` (I→I / T→I Recall@K + mAP,
베이스라인 비교 지원).

### ③ CLIP InfoNCE의 false negative [구현됨]
특허 제목은 "Sneaker"처럼 짧고 중복이 많음. 배치 안에 같은 디자인의 다른 뷰나
동일 텍스트가 들어오면 표준 InfoNCE가 "정답인데 네거티브로 밀어내는" 노이즈 발생.
PK 샘플러 도입 시 같은 디자인 뷰들이 한 배치에 모이므로 **이 마스킹 없이는 PK
샘플러가 오히려 해로움** — 둘은 반드시 세트.

**해결**: HF 내장 `return_loss` 대신 직접 구현한 **멀티-positive InfoNCE**
(`masked_clip_loss`) — 같은 `design_id`의 다른 뷰를 positive로 처리.
단 **텍스트 동일성은 positive 근거에서 제외**했다: 고유 제목 28,859개 중 유일한 것이
141개뿐이라(2026-08 실측) 제목으로 묶으면 서로 다른 디자인이 정답이 되고, 특히
`locarno_aware` 배치에서 마스크가 거의 전부 True가 되어 변별 그래디언트가 사라진다.

### ④ 텍스트 빈약 [부분 구현]
text가 `object_title` 하나뿐. 개선 수단:
- `build_pairs.py --include-aspect`로 뷰포인트 부가 (기존 옵션, 활용 권장)
- 학습 중 확률적 viewpoint 부가 (Collator `augment=True` 시 30% 확률) [구현됨]
- **한국어 병기 텍스트**: 목표가 한국어 검색인데 학습 텍스트가 전부 영어 →
  한국어 성능이 베이스 모델의 다국어 정렬에 전적으로 의존. 제목 번역 레코드를
  섞으면 직접적 개선. **[미구현 — 번역 파이프라인 필요, 로드맵 참조]**
- 로카르노 분류 명칭 부가(`data/locarno_9.json` 활용) **[미구현]**

### ⑤ 학습 증강 미구현 [구현됨]
DESIGN.md §3에 설계만 있고 구현이 없었음. `preprocess_drawing(augment=True)`로
소회전(±7°)·콘텐츠 스케일(0.85~1.0)·랜덤 배치 오프셋·라인 두께 변화
(morphology Min/MaxFilter)를 추가. eval/추론은 기존 결정적 전처리 유지.

### ⑥ 검색 결과 집계가 max-only [구현됨]
서버 `_dedup_pack`이 출원별 **최고 점수 뷰 1장**만 남김 → 노이즈에 취약.
**해결**: 출원별 상위 2개 뷰 점수 평균으로 집계(`_group_pack`), 대표 이미지는
최고 점수 뷰 유지.

### ⑦ 재정렬(re-rank) 부재 [구현됨 — αQE]
**해결**: α-쿼리 확장(αQE) — 1차 검색 top-m 이미지 벡터를 점수^α 가중 평균해
쿼리를 보강 후 재검색. 랜드마크 리트리벌 표준 기법, 코드 수 줄로 mAP 개선.
k-reciprocal re-ranking은 더 강력하지만 구현 복잡도가 높아 로드맵으로 이관.

### ⑧ 쿼리 단일 인코딩 [구현됨]
- 텍스트: 프롬프트 템플릿 앙상블(한글 감지 시 한국어 템플릿) 평균
- 이미지: TTA(원본 + ±4° 회전) 평균

### ⑨ 로카르노를 스코어에 미반영 [미구현 — 로드맵]
파셋 표시에만 사용 중. 예측 카테고리와 로카르노가 일치하는 후보 가점 등.

### ⑩ 유효 배치 32로 제한 [미구현 — 로드맵]
대조학습은 네거티브 수(배치)에 민감. GradCache로 유효 배치 수백까지 확대 가능.
대안: SigLIP식 sigmoid loss(작은 배치에 강건).

### ⑪ 해상도 224 [미구현 — 로드맵]
라인 드로잉은 얇은 선이 정보의 전부. `-378`/`-s16-384` 변형 실험 가치 있음.

### ⑫ LoRA 범위 [미구현 — 로드맵]
`target_modules`에 `fc1,fc2` 추가, `train_projections: true`, rank 16→32.
데이터 확장 후 시도 권장.

### ⑬ 사진 쿼리 도메인 갭 [미구현 — 로드맵]
웹앱은 임의 이미지를 받지만 학습은 도면뿐. 실사용에 제품 **사진** 쿼리가 있다면
자연 이미지→에지맵(HED/Canny) 합성 쌍 학습 고려.

---

## 2. 구현 매핑

| # | 항목 | 파일 / 함수 | 온오프 |
|---|------|-------------|--------|
| ① | PK 배치 샘플러 (+로카르노 하드 네거티브) | `src/dataset.py` `PKBatchSampler` | `train.pk_views` (1=비활성), `train.locarno_aware` |
| ② | design_id 분할 | `src/dataset.py` `split_by_design` → `train.py` | 항상 |
| ② | 전체 갤러리 평가 (R@K, mAP) | `src/eval_retrieval.py`, `src/metrics.py` | CLI |
| ③ | false-negative 마스킹 InfoNCE | `src/train.py` `masked_clip_loss` | `train.mask_false_negatives` |
| ④ | viewpoint 확률 부가 | `src/dataset.py` `Collator(augment=True)` | `train.augment` |
| ⑤ | 도면 증강(회전·스케일·라인두께) | `src/dataset.py` `preprocess_drawing(augment=)` | `train.augment` |
| ⑥ | 출원별 top-2 뷰 평균 집계 | `webapp/server.py` `_group_pack` | 항상 |
| ⑦ | αQE 쿼리 확장 | `webapp/server.py` `_query_expand` | 요청 파라미터 `qe` (기본 on) |
| ⑧ | 텍스트 템플릿 앙상블 / 이미지 TTA | `webapp/server.py` `_encode_text_query` / `_encode_image_query` | `search_text` 항상 / 요청 파라미터 `tta` (기본 on) |
| — | 베이스라인 평가(어댑터 없이) | `src/embed.py` `load_tuned(adapter="none")` | CLI |

부가 수정: `train.py`의 배치 내 평가(`evaluate`)도 멀티-positive 인식으로 교정
(같은 `design_id`의 다른 뷰가 top-K에 있으면 히트 — 손실과 같은 `_pos_mask`를 쓰므로
텍스트 동일성은 보지 않는다). 단, 이는 학습 중 빠른 프록시일 뿐이며 **공식 수치는
`eval_retrieval.py`가 기준**.

---

## 3. 실행 가이드 (학습 PC)

```powershell
.\.venv\Scripts\Activate.ps1

# 0) 파이프라인 스모크 (모델 로드/LoRA/forward)
python scripts\verify_pipeline.py
python src\train.py --limit 512 --max-steps 10 --eval-batches 2   # 학습 루프 스모크

# 1) 학습 전 베이스라인 측정 (전체 갤러리, design 분할 홀드아웃)
python src\eval_retrieval.py --adapter none

# 2) 재학습 (PK 샘플러 + 마스킹 + 증강은 config 기본값으로 활성)
python src\train.py --config configs\lora_clip.yaml

# 3) 튜닝 후 평가 → 베이스라인과 비교
python src\eval_retrieval.py --adapter outputs\lora-clip-design\final

# 4) 인덱스 재구축 + 서버
python src\embed.py build --adapter outputs\lora-clip-design\final --data data\pairs.jsonl
python webapp\server.py
```

주의:
- **기존에 저장된 eval 수치와 새 수치는 비교 불가** (분할 방식이 바뀌어 홀드아웃
  자체가 다름). 베이스라인부터 다시 측정할 것.
- `eval_retrieval.py`는 `--eval-ratio`/`--seed`가 학습 config와 같아야 같은
  홀드아웃을 봄 (기본값이 config와 동일하게 맞춰져 있음).
- 로카르노 클래스 내 동일 명칭 비율이 높으면 PK 배치에서 텍스트 대비 신호가
  줄어들 수 있음 → `build_pairs.py --include-aspect`로 텍스트 차별화 권장.

---

## 3-1. 방법별 기여도 측정 (Ablation)

각 개선 방법이 **개별적으로 얼마나 성능을 올리는지** 측정하는 자동 러너:
`scripts/run_ablation.py`. arm(실험군)마다 **학습 → 전체 갤러리 평가 → FAISS 인덱스
빌드** 3단계를 돌고 `outputs/methods/summary.md`에 baseline 대비 Δ 비교표를 만든다.
모든 arm이 같은 seed → **동일한 design_id 홀드아웃**에서 비교됨.

방법 정의는 `configs/methods/<name>.yaml` 한 장이 규정하고, 병합은 `src/registry.py`
한 곳에서만 한다(새 arm은 YAML 한 장 추가로 끝). 병합 결과는 arm마다
`outputs/methods/<name>/config.resolved.yaml`로 못박히고 학습·평가·인덱스가 이 파일만
읽는다 — 세 단계가 같은 데이터·해상도를 본다는 보장이 여기서 나온다.

```
outputs/methods/<name>/
  config.resolved.yaml   ← 재현 기록. 학습을 스킵할 때는 덮어쓰지 않는다
  data/pairs.jsonl.pointer.json
  final/                 ← LoRA 어댑터
  eval/final.json  ·  index/{faiss.index, meta.jsonl, index_meta.json}
  train.log · eval.log · index.log
```

| arm | 구성 | 측정 대상 |
|---|---|---|
| `base` | 튜닝 전 베이스 모델 (평가만) | LoRA 튜닝 자체의 효과 기준점 |
| `baseline` | LoRA 학습, 모든 개선 OFF | 기존 파이프라인 |
| `aug` | +증강 | ⑤ 단독 기여 |
| `mask` | +마스킹 | ③ 단독 기여 |
| `pkmask` | +PK 샘플러+마스킹 | ①+③ (세트 — PK 단독은 유해) |
| `pkmask-i2i` | +PK+마스킹+supcon | ①+③+img2img loss |
| `all` | 전부 ON | 최종 레시피 |
| (선택) `hobit` | baseline + submodular greedy 배치 구성 | ① 배치 구성의 원리적 대체 (ICML 2026) |
| (선택) `tic` | baseline + 텍스트 모달 내부 대조(헤드명사+상한 선택) | ③ 같은 물품군의 다른 물품이 텍스트 공간에서 구분되지 않음 |
| (선택) `pk-only` | PK만, 마스킹 없이 | 유해 상호작용 실증 |
| (선택) `all-mlp` / `all-proj` / `all-r32` | all + LoRA 확장 | ⑫ 계열, 최종 레시피 위 추가 기여 |
| (선택) `loracap` | baseline + LoRA 용량 확장(rank 32, fc1/fc2, projection) | ⑫ 파라미터 용량 축의 단독 기여 (학습 파라미터 4.5배, 표에서 가장 비싼 arm) |

`(선택)` 표시가 없는 행(`base`~`all`)만 `run_ablation.py`를 인자 없이 돌렸을 때 학습된다
(= `DEFAULT_ARMS` + 항상 앞에 붙는 `base`). `(선택)` 행은 `--arms`로 이름을 적어야 돈다 —
`hobit`·`tic`·`loracap`도 여기 속한다(각각 배치 구성 비용, 미검증 하이퍼파라미터,
학습 파라미터 4.5배로 인한 VRAM·스텝 비용 때문).

```powershell
python scripts\run_ablation.py --quick        # 스모크 (레코드 2000 × 1 epoch)
python scripts\run_ablation.py --epochs 3     # 실전 ablation (권장 시작점, 학습 6회)
python scripts\run_ablation.py --report-only  # 완료된 결과로 표만 재생성
python scripts\run_ablation.py --arms baseline all all-r32 --epochs 3   # arm 선택
python scripts\run_ablation.py --epochs 3 --no-index   # 인덱스 빌드 생략 (평가만 볼 때)
python scripts\run_ablation.py --arms baseline hobit --epochs 3   # 배치 구성 단독 기여
python scripts\run_ablation.py --arms baseline tic --epochs 3   # 손실 축 단독 기여
python scripts\run_ablation.py --arms baseline loracap --epochs 3   # 파라미터 용량 축 단독 기여
```

- `hobit` arm은 다른 arm보다 비싸다: 에폭마다 학습 분할 전체(425,140장)를 1회 추론해
  배치 구성용 임베딩을 갱신한다(실측 디코딩 98.6분 + ViT-H forward 27.9분 → 디코딩을
  `num_workers`로 병렬화해도 에폭당 수십 분 추가). 비용을 줄이려면 메서드 YAML의
  `hobit_refresh_every`를 2 이상으로 (N 에폭마다 1회 갱신).
- `loracap` arm은 학습 파라미터가 4.5배(8,388,609 → 37,888,001)라 활성값도 같이 커진다.
  batch 32 실측(RTX 5090 31.84GiB, bf16, 224px, 제목이 긴 배치 = 토큰 77 기준):
  체크포인팅을 끄면 peak 32.03GiB로 VRAM을 넘겨(여유 0.00GiB) 시스템 메모리로 스필한다:
  크래시도 경고도 없이 0.28 → 4.05 s/step(약 14배)이 되고, 13,285 step/epoch 기준
  `--epochs 3`이 약 6시간에서 약 45시간이 된다. 그래서 메서드 YAML에
  `gradient_checkpointing: true`가 켜져 있다(활성값을 저장하는 대신 backward에서 재계산
  → peak 10.45GiB, 0.53 s/step, 여유 21.4GiB). 남는 비용은 baseline 대비 약 1.9배
  (0.28 → 0.53 s/step)다. batch_size를 줄여 VRAM을 맞추지 않은 이유는 in-batch 네거티브
  수까지 같이 줄어 Δ를 용량 축의 기여로 읽을 수 없게 되기 때문이다(해상도 축을 미룬 것과
  같은 이유).
- 같은 이유로 `all-mlp`/`all-r32`도 batch 32에서 스필한다(fc1/fc2 활성값이 지배적).
  이 브랜치에서는 손대지 않았다 (후속 작업에서 같은 플래그를 켜면 된다).
- 학습 1회가 오래 걸리므로 `--epochs 3`으로 좁힌 뒤, 유망한 조합만 풀 epoch 재학습 권장.
- 중단돼도 재실행하면 완료된 arm(어댑터/평가/인덱스 존재)은 자동 스킵 (`--force`로 무시).
- 인덱스 빌드는 `--limit`을 포함한 지문(`index/index_meta.json`)을 남긴다. `--quick`으로
  만든 2,000장짜리 데모 인덱스를 나중에 전체 갤러리로 착각해 서빙하는 사고를 막는다
  (`src/embed.py`의 `check_index_meta`가 판정).
- 어댑터가 이미 있는 arm에서 메서드 YAML을 고치고 `--force` 없이 재실행하면, 저장된
  `config.resolved.yaml`을 **그대로 쓰고** 달라진 키를 경고로 출력한다. 새 설정으로
  돌리려면 `--force`.

## 3-2. 관련 논문 (ICML 2026 로컬 포스터 아카이브, localhost:8000)

5,862편 중 프로젝트 관련 키워드 검색으로 선별 (ID = 아카이브/icml.cc 포스터 번호).

**학습 방법 — 본 문서 ①③⑨와 직결:**
- **HOBIT: Hardness Optimized Batch Sampling for InfoNCE Training** [64085]
  — InfoNCE용 배치 구성 최적화(하드하되 모순 없는 네거티브). 우리 PK+로카르노
  휴리스틱 배치의 원리적 대안/보강. **가장 직접적 관련.**
- **FG-CLIP 2: Bilingual Fine-grained Vision-Language Alignment** [65426]
  — 이중언어 fine-grained CLIP + TIC loss(유사 캡션 구분). 한국어 병기(④)와
  동일 명칭 다수 문제(③)에 시사점. 가중치 공개 시 베이스 모델 후보.
- **IN²R: Rectifying Inter-Modal Noisy Correspondence** [65301]
  — 노이즈 캡션을 배제 대신 모달 내 이웃 그래프로 소프트 타깃 합성.
  특허 제목의 빈약/노이즈 대응(③의 확장).
- **VACSR: Variational Adapter for Cross-modal Similarity** [65621]
  — 이진 매칭 라벨이 만드는 false negative를 변분 추론으로 해소. ③의 이론적 근거.
- **SOLAR: Symmetric Multimodal Retrieval** [61620]
  — 대칭 검색(쿼리↔갤러리 교환 가능) + 마스킹 기반 hard negative 자가 생성.
  우리 양방향(T→I, I→I) 설정과 유사한 문제 정의.

**검색/서빙 단계 — ⑥⑦⑧⑨와 관련:**
- **SEPS: Semantic-Enhanced Patch Slimming** [61362] — 전역 평균 풀링의 유사도
  희석을 salience 기반 집계로 해결. top-K 후보 패치 단위 re-rank 아이디어.
- **Pix2Key: Controllable Open-Vocabulary Retrieval** [60716] — composed image
  retrieval(참조 이미지+텍스트 수정). 우리 '개념 검색'(이미지+카테고리 융합)의 발전형.
- **ReEx: Robust Cross-modal Retrieval via TTA** [60622] — 노이즈 쿼리 스트림에
  대한 테스트타임 적응. 실사용 쿼리(사진·오타) 강건성(⑬).
- **LCSE: Similarity Is Not Logic** [65419] — 학습 없이 부정/조합 쿼리("A인데 B 없는")
  를 스코어 편집으로 처리. 듀얼 인코더의 bag-of-concepts 한계 우회.

**배경 인사이트:**
- **How can embedding models bind concepts?** [61962] — CLIP이 속성-객체 결합에
  실패하는 원인 분석. fine-grained 도면 검색의 근본 한계 이해용.
- **ObjEmbed** [63895] — 영역 단위 임베딩. 부분 형상 검색이 목표가 되면 참조.

## 4. 남은 로드맵 (우선순위순)

1. **④ 한국어 텍스트 증강**: object_title을 한국어로 번역(LLM 배치 번역)해
   같은 이미지에 en/ko 레코드 병기 → 한국어 쿼리 정확도 직접 개선.
2. **⑨ 로카르노 스코어 반영**: 검색 시 예측 카테고리·로카르노 일치 후보 가점.
3. **⑦+ k-reciprocal re-ranking**: αQE보다 강력한 상호 이웃 기반 재정렬.
4. **⑩ GradCache**: 유효 배치 32 → 256+ 확대. 또는 sigmoid loss 실험.
5. **⑪ 해상도 384** 변형 실험 (`image_size`+`model_id` 변경만으로 가능).
6. **⑫ LoRA 범위 확장**: `fc1,fc2` + projections + rank 32.
7. **⑬ 사진→도면 쿼리**: 에지맵 합성 쌍 또는 쿼리 전처리(에지 추출) 실험.
