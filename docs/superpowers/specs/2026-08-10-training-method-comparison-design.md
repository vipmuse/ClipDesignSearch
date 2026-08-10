# 학습 방법 비교 구조 설계

작성 2026-08-10. 대상 저장소 `ClipDesignSearch` (MetaCLIP 2 + LoRA 디자인/특허 도면 검색).

5개 학습 방법을 같은 조건에서 학습·평가하고, 웹 검색 페이지에서 나란히 비교한다.
방법을 1급 개념으로 올려 데이터 준비부터 서빙까지 하나의 레지스트리로 규정한다.

---

## 1. 배경과 근거

### 1.1 데이터 실측

```
472,615 레코드 / 60,784 디자인 / 고유 제목 28,859개
최빈 제목: Shoe 4,720 · Bottle 3,771 · Container 3,741 · Mobile phone 3,603
단 한 번만 등장하는 제목: 141개
로카르노 코드: 전 레코드 보유, 271종 사용
```

### 1.2 선행 결함 — baseline 정상화가 전제

`src/train.py`의 `_pos_mask`는 **텍스트가 같으면 positive**로 표시한다. 고유 제목
28,859개 중 유일한 것이 141개뿐이므로 이 조건은 거의 항상 발동한다. 여기에
`locarno_aware: true`로 같은 로카르노 클래스에서 배치를 뽑으면 배치 32개가 모두
"Shoe"가 되어 `pos_mask`가 사실상 전부 True가 되고, `masked_clip_loss`는 변별
그래디언트를 잃는다. 두 옵션 모두 `configs/lora_clip.yaml` 기본값이 ON이다.

또한 `--limit`이 `random.shuffle` 없이 앞에서부터 잘라내므로 축소 실험이 인제스천
순서상 한 구간(특허번호·로카르노가 몰린 표본)만 본다.

**이 두 가지를 고치기 전의 비교는 망가진 바닥 위에서 이뤄진다.** 따라서 본 작업의
선행 단계로 포함한다 (§6 Phase 0).

### 1.3 HOBIT의 위치

HOBIT (ICML 2026 Spotlight, Dutta·Nagalapatti·Prabhu)은 에폭마다 학습 예제를
재정렬해 각 쿼리가 "hard하되 모순되지 않는" 네거티브를 보도록 미니배치를 구성한다.
목적함수가 monotone submodular임이 증명되어 greedy 알고리즘이 (1−1/e) 근사를 보장한다.

이는 현재 `PKBatchSampler` + `locarno_aware` 휴리스틱이 하려던 일의 원리적 대체이며,
§1.2가 지적한 "contradictory negative" 문제를 손실 함수가 아니라 **배치 구성 단계**에서
다룬다.

> **한계 명시**: OpenReview(`R49XZi14YH`)가 브라우저 차단을 걸고 arXiv 프리프린트가
> 없어 **전문을 확보하지 못했다.** 확보된 것은 구조(에폭 단위 재정렬 · submodular ·
> greedy · 근사 보장)뿐이고 정확한 hardness 스코어 함수와 하이퍼파라미터는 없다.
> 따라서 본 구현은 **논문 재현이 아니라 원리 차용**이다. 스코어 함수는 §3.1의 정의를
> 쓰고, 논문 입수 시 교체 가능하도록 `hobit_score` 선택자로 분리한다.

---

## 2. 아키텍처 — 메서드 레지스트리

방법 하나를 `configs/methods/<name>.yaml` 한 장이 완전히 규정한다.

```yaml
name: hobit
description: "에폭마다 submodular greedy 재정렬로 배치 구성"
extends: configs/lora_clip.yaml
data:
  builder: shared          # shared | (향후 korean, edge …)
  pairs: data/pairs.jsonl
model: {}                  # hires384만 model_id/image_size 오버라이드
train:
  sampler: hobit           # random | pk | hobit
  hobit_pool: 4096
  hobit_score: cosine
```

산출물은 방법마다 동일한 모양을 갖는다.

```
outputs/methods/<name>/
  config.resolved.yaml   ← extends 병합 결과. 이후 단계는 전부 이것만 읽는다
  data/pairs.jsonl       ← builder: shared면 공용 파일을 가리키는 포인터 파일
  adapter/final/
  index/{faiss.index, meta.jsonl, index_meta.json}
  eval/final.json
  train.log, eval.log
```

**설계 원칙: 병합은 한 곳에서만.** 학습·평가·인덱스·서빙이 각자 YAML을 병합하면 네
곳의 병합 로직이 어긋난다. 레지스트리 로더(`src/registry.py`)만 병합하고 결과를
`config.resolved.yaml`로 못박는다. 이후 단계는 이 파일만 읽는다. 재현성도 여기서 나온다.

**포인터 파일**을 쓰는 이유: 472k 레코드를 방법마다 복제하지 않기 위해서고, Windows에서
심볼릭 링크는 관리자 권한을 요구하기 때문이다. `{"source": "data/pairs.jsonl"}` 형태.

`data.builder` 필드는 5개 방법이 모두 `shared`인 지금도 유지한다. 향후 에지맵 합성이나
번역 계열을 추가할 때 구조를 다시 뜯지 않기 위한 자리다.

### 2.1 러너 통합

`scripts/run_ablation.py`의 `ARMS` 딕셔너리를 `configs/methods/*.yaml`로 이관한다.
기존 arm(baseline/aug/mask/pkmask/pkmask-i2i/all)도 같은 형식의 YAML이 되므로 러너는
하나로 유지된다. 러너는 방법마다 **학습 → 평가 → 인덱스 빌드** 3단계를 돈다.
인덱스 빌드가 새로 붙는 단계다 (현재 ablation은 어댑터와 평가까지만 만든다).

---

## 3. 5개 방법

모두 정상화된 baseline 위에 **한 가지 축만** 변경한다. 기여도가 섞이지 않게 하기 위함이다.

| 방법 | 바꾸는 축 | 구현 위치 | 선정 근거 |
|---|---|---|---|
| `hobit` | 배치 구성 | `src/dataset.py` 샘플러 | 필수 지정. §1.3 |
| `tic` | 텍스트 모달 내부 대조 | `src/train.py` 손실 | 제목 중복 실측(141/28,859)이 이 데이터의 핵심 난점 |
| `bigbatch` | 네거티브 수 | `src/train.py` GradCache | 대조학습의 배치 크기 민감성 |
| `hires384` | 입력 해상도 | `model.model_id` + `image_size` | 라인 드로잉은 얇은 선이 정보의 전부 |
| `loracap` | 파라미터 용량 | `lora.*` | 저비용 용량 확장 |

데이터는 5개 모두 `shared`다.

### 3.1 `hobit`

에폭 시작 시점에 현재 모델로 학습 부분집합의 이미지 임베딩을 계산하고, 그 임베딩으로
hardness를 정의해 greedy로 배치를 채운다.

- **hardness**: 후보와 이미 배치에 담긴 예제들 사이 코사인 유사도의 합 (높을수록 hard).
- **contradiction 페널티**: 같은 `design_id`이거나 동일 텍스트인 쌍은 감점. 이것이
  "non-contradictory"에 해당하며 §1.2 문제의 배치 단계 대응이다.
- **greedy**: 배치가 빌 때까지 (hardness − 페널티)가 최대인 후보를 하나씩 넣는다.
  전체 472k에 대한 완전 탐색은 불가능하므로 `hobit_pool`(기본 4096) 크기의 무작위
  후보 풀에서 고른다.
- 에폭당 모든 예제가 **정확히 한 번** 배치에 들어가야 한다 (§5 테스트 대상).

기존 `PKBatchSampler`는 `sampler: pk`로 남겨 비교 대상으로 유지한다.

### 3.2 `tic`

FG-CLIP 2의 Textual Intra-modal Contrastive 계열. 배치 안에서 **같은 물품군에 속하지만
서로 다른 물품**인 제목 쌍을 텍스트 공간에서 밀어낸다. `Pizza box`와 `Storage box`가
붙어 있으면 텍스트 검색이 둘을 가르지 못한다.

**선택 규칙은 두 축을 함께 쓴다 (2026-08-11 실측으로 확정).**

1. **같은 헤드명사** (제목의 마지막 단어) — 관련 물품군으로 좁힌다.
2. **코사인 < 상한 0.92** — 같은 물품의 표기 차이를 제외한다.
3. 문자열이 같은 쌍, 같은 `design_id` 쌍 제외.

밀어내는 강도는 바닥값(floor) 위의 hinge: `(cos − floor).clamp(min=0)`, floor는 같은
헤드명사 쌍 분포의 중앙값 0.75.

> **왜 스칼라 임계값 하나로는 안 되는가 — 실패한 1차 설계의 기록.**
> 초안은 "코사인이 margin(0.9)보다 높은 쌍을 밀어낸다"였고, 이는 실측으로 반증됐다.
> 베이스 모델로 잰 코사인은 설계가 가정한 순서로 정렬되지 않는다:
> `Container`/`Beverage container` 0.786, `Shoe`/`Ballet shoe` 0.801 — 목표 쌍인데
> margin 아래라 절대 발동하지 않는다. 반면 무관한 `Shoe`/`Bottle`은 0.867로 그보다 높고,
> margin을 넘는 것은 `Eyeglasses`/`Glasses` 0.951, `Wash basin`/`Washbasin` 0.968처럼
> **붙어 있어야 맞는** 쌍들이다. 그 결과 TIC 항은 CLIP 손실 3.5에 대해 4e-6을 기여했다
> (배치당 대상 496쌍 중 0.35쌍만 발동). 가중치를 올리면 동의어를 떼어놓고, margin을 내리면
> 전체 쌍의 30%가 들어와 일반 분산 정규화로 변질된다. 하나의 스칼라에 두 가지 일(물품군
> 선택 + 표기차 배제)을 시킨 것이 원인이었다.

**실측 근거** (베이스 모델, 같은 헤드명사 쌍 40만 표본):

```
분포: p50 0.765 · p75 0.824 · p90 0.870 · p95 0.896 · p99 0.950 · ≥0.92는 2.45%
상한이 걸러낼 최상위: 'Portable Bluetooth speaker'/'Portable bluetooth speaker' 0.997,
  'Clothing hanger'/'Clothes hanger' 0.996, 'Handheld'/'Hand-held vacuum cleaner' 0.994
  → 전부 같은 물품의 표기 차이
0.78~0.88 구간(목표군): 'Wine carrier'/'Pet carrier', 'Pizza box'/'Storage box',
  'Button panel'/'Wall panel', 'Flow sensor'/'Sensor', 'Vehicle seat'/'Collapsible seat'
배치(32) 대상 쌍: 평균 3.03 (중앙값 2, 최대 16), 0개인 배치 10.3%
  — 기존 규칙 495.7쌍 대비 165배 감소. .mean()의 분모가 줄어 쌍당 가중치가 그만큼 커진다.
```

**가중치는 실측 후 정한다.** CLIP 항은 `logit_scale` 증폭을 받지만 hinge는 받지 않아,
전체 쌍이 위반하는 극단에서도 `tic_weight 0.2`의 그래디언트 비는 0.004에 그쳤다. 첫 실행은
반드시 `--max-steps 5 --log-every 1`로 돌려 로그의 TIC 항·대상 쌍 수·위반 쌍 수를 보고
가중치를 정한다 (로깅은 구현되어 있다).

### 3.3 `bigbatch`

GradCache로 유효 배치를 32 → 256으로 확대한다. 두 번의 forward(그래디언트 없이 임베딩
캐시 → 재계산)로 메모리를 상수로 유지한다. `batch_size`는 32로 두고
`grad_cache_chunks: 8`로 유효 배치를 만든다. VRAM 실측(batch 64 = peak 37.5GB > 32GB)상
물리 배치를 키우는 길은 막혀 있다.

### 3.4 `hires384` / 3.5 `loracap`

`hires384`는 `model_id`를 고해상도 변형으로, `image_size`를 그에 맞게 바꾼다. 임베딩 공간
자체가 달라지므로 인덱스는 반드시 재빌드되며, 다른 방법과 백본을 공유할 수 없다 (§4.1).

> **반드시 `huge-378`을 쓴다.** `configs/lora_clip.yaml` 주석은 `s16-384`도 후보로 적고
> 있으나, s16은 백본 크기 자체가 huge와 다르다. 이를 쓰면 해상도와 파라미터 용량이 함께
> 바뀌어 "한 축만 변경" 원칙이 깨지고 `loracap`과 기여도가 뒤섞인다. 해상도 축을 분리하려면
> 백본 계열을 huge로 고정해야 한다. 로컬에는 `huge-quickgelu`(224)만 있으므로
> `scripts/download_model.py --model-id facebook/metaclip-2-worldwide-huge-378`로 추가
> 다운로드가 필요하다 (2026-08 확인: 해당 저장소 존재, 원 구현의 ViT-H-14-378-worldwide).
> `image_size`는 378로 맞춘다 — 384가 아니다.

`loracap`은 `target_modules`에 `fc1,fc2` 추가, `train_projections: true`, `r: 32`/`alpha: 64`.

---

## 4. 서빙

### 4.1 모델 적재

`hobit`/`tic`/`bigbatch`/`loracap`은 `model_id`가 같으므로 **베이스 백본 하나를 공유**하고
PeftModel의 named adapter로 `set_adapter(name)` 전환한다. 현재 `load_tuned(merge=True)`는
어댑터를 백본에 병합하므로 다중 어댑터와 양립하지 않는다 — 비교 서버에서는
`merge=False` 경로를 쓴다. `hires384`만 `model_id`가 달라 별도 백본을 올린다.

### 4.2 RAM 예산

```
시스템 RAM 33.4 GB (가용 18.3 GB) · 인덱스 1개 1.94 GB (472,615 × 1024 × 4B)
```

현재 `server.py`는 인덱스를 올린 뒤 `index.reconstruct_n(0, ntotal)`으로 전체 벡터를
**한 벌 더** 복사한다(server.py:52). 방법당 1.94 + 1.94 + meta ≈ 4.2 GB, 5개면 21 GB로
가용 RAM을 넘긴다. 세 가지로 해결한다.

1. **중복 벡터 캐시 제거** — `IndexFlatIP`가 이미 원본 벡터를 보유하므로 복사본은 낭비다.
   `STATE["vectors"][ids]` 사용처(`_query_expand`, 개념 검색 재정렬)를
   `index.reconstruct_batch(ids)`로 교체한다. 쿼리당 수백 벡터만 꺼내므로 비용은 무시할 수준.
2. **meta 공유** — 5개 방법이 같은 레코드 집합을 인덱싱하므로 meta는 한 벌만 두고 방법별
   id 매핑만 유지한다.
3. **부분집합 인덱스** — §4.3.

### 4.3 실험 규모

학습 비용이 실질적 병목이다: 472k ÷ batch 32 = 14,769 스텝/epoch, 10 epoch이면 방법당
14.8만 스텝으로 ViT-H에서 며칠이 걸린다. 5개는 불가능하다.

**2단계 프로토콜**을 쓴다.

- **비교 단계**: 로카르노 코드로 계층화 추출한 `design_id` 단위 부분집합으로 5개 모두
  학습·인덱싱. 목표 규모는 **도면 약 100k = 디자인 약 12,900개**(전체 평균 7.8 뷰/디자인
  기준). 디자인 단위로 뽑아 한 디자인의 뷰가 쪼개지지 않게 한다. 인덱스 0.41 GB × 5 =
  약 2 GB로 RAM에 여유가 있고 학습도 5배 빠르다. 부분집합은 `data/subset_100k.jsonl`로
  고정 seed 생성해 모든 방법이 동일 표본을 본다.
- **승격 단계**: 비교에서 이긴 방법만 472k 전체로 재학습·재인덱싱해 운영에 쓴다.

### 4.4 API

```
POST /api/search_multi   (image, topk, methods=[…])
POST /api/search_text_multi
GET  /api/methods        → 활성/미생성 방법 목록과 각 지표
```

응답은 방법별 결과와 **합집합 뷰**를 함께 담는다. UI(§4.5)가 후자를 쓴다.

```json
{
  "methods": [{"name": "hobit", "results": [...], "error": null}, ...],
  "union": [{"idx": 123, "image_url": "...", "picks": {"hobit": 1, "tic": 1, "bigbatch": 3}}]
}
```

### 4.5 UI — 합집합 그리드 + 방법 배지

브라우저 목업 3안(방법별 세로 컬럼 / 합집합 그리드 / 기준 대비 차이) 중 **합집합 그리드**를
채택했다.

모든 방법의 결과를 중복 없이 한 그리드에 놓고, 각 도면 카드에 "어떤 방법이 몇 위로
뽑았는지"를 배지로 단다. 다수 방법이 공통 선택한 도면과 한 방법만 찾아낸 도면을 색으로
구분한다. 방법별 세로 컬럼은 같은 도면이 5번 반복돼 정작 *차이*가 묻히는 반면, 합집합
그리드는 화면이 곧 차이 지도가 된다.

배지를 클릭하면 그 방법만 필터링해 "이 방법의 top-N"을 볼 수 있게 한다 — 컬럼 방식이
갖던 가독성을 보완한다.

### 4.6 실패 격리 — 부분 가용성이 기본

5개 방법이 동시에 준비되는 일은 없다. 학습은 순차로 며칠에 걸쳐 끝나고 그 사이에도
웹앱은 돌아가야 한다.

- 기동 시 `outputs/methods/*`를 스캔해 **인덱스가 완성된 방법만 활성**으로 올린다. 없는
  방법은 UI에 회색 "미생성"으로 표시하고, 하나도 없으면 기존 단일 인덱스로 폴백한다.
- `index_meta.json`의 지문(방법명·어댑터·데이터·`model_id`·`image_size`·차원)이 실제 로드한
  어댑터와 어긋나면 **그 방법만 비활성**시킨다. 서버 전체를 죽이지 않는다. 방법이 5개로
  늘면 "hobit 인덱스에 tic 어댑터로 검색하는" 사고가 실제로 일어나고, 그건 에러 없이
  결과만 조용히 틀어진다. `src/embed.py`의 `ckpt_key`와 같은 원칙이다.
- 검색 시 한 방법이 예외를 던져도 나머지 결과는 반환한다. 방법별 `error` 필드를 두고 UI는
  그 부분만 실패 표시한다.
- 기동 시 `활성 방법 수 × 인덱스 크기`를 계산해 가용 RAM을 넘으면 경고하고 초과분을
  비활성으로 둔다.

---

## 5. 테스트

저장소에 테스트가 없다(`pytest` 미설치). 전면 도입이 아니라 **이 기능이 조용히 틀어지는
지점만** 겨냥해 `tests/`를 추가한다.

- **레지스트리 로더** — `extends` 병합이 결정적이고 `config.resolved.yaml`이 실제 학습에
  쓰인 값과 일치하는가. 진실원천이 하나라는 전제가 깨지면 나머지가 무의미해진다.
- **HOBIT 샘플러** — 에폭당 모든 예제가 정확히 한 번 들어가는가, 같은 `design_id`가 한
  배치에 한도 이상 들어가지 않는가, 같은 seed에서 재현되는가. greedy 구현은 조용히 예제를
  누락하거나 중복시키기 쉽다.
- **positive 마스크 회귀** — 배치에 "Shoe" 제목이 32개 있어도 `pos_mask`가 대각선과 같은
  `design_id`만 True인지 단언한다. §1.2의 재발 방지.
- **`index_meta` 불일치 검출** — 어댑터를 바꿔치기하면 그 방법이 비활성되는가.
- **소규모 e2e 스모크** — 100 레코드 × 2 방법으로 학습 → 인덱스 → 검색이 끝까지 도는가.

---

## 6. 구현 순서

- **Phase 0 — baseline 정상화**: `_pos_mask`를 `design_id` 기준으로, `--limit` 전 셔플 복원.
  이것 없이는 이후 비교가 무의미하다.
- **Phase 1 — 레지스트리**: `src/registry.py`, `configs/methods/*.yaml`, 기존 `ARMS` 이관,
  `config.resolved.yaml` 산출. 테스트 동반.
- **Phase 2 — 인덱스 파이프라인**: 러너에 인덱스 빌드 단계 추가, `index_meta.json` 기록,
  부분집합 `data/subset_100k.jsonl` 생성기.
- **Phase 3 — 방법 구현**: `hobit` → `tic` → `bigbatch` → `loracap` → `hires384`.
  각 방법은 독립이므로 순서를 바꿔도 되고, HOBIT을 먼저 해서 파이프라인을 검증한다.
- **Phase 4 — 서빙**: 다중 어댑터 적재, RAM 절감(중복 캐시 제거·meta 공유),
  `/api/search_multi`, 부분 가용성.
- **Phase 5 — UI**: 합집합 그리드 + 방법 배지 + 배지 필터.

Phase 0~2는 방법 구현과 무관하게 선행되어야 하고, Phase 4~5는 방법이 하나만 완성돼도
동작을 확인할 수 있다.

**계획 분할**: Phase 0~2(기반)와 Phase 4~5(서빙·UI)는 각각 하나의 구현 계획으로 묶기에
적당하다. Phase 3의 5개 방법은 서로 독립이고 각각 알고리즘 구현이 필요하므로 방법마다
별도 계획으로 다룬다. 즉 이 스펙은 계획 하나가 아니라 계획 여러 개의 근거 문서다.

---

## 7. 미결 사항

### 7.1 Phase 0~2 구현 후 남은 후속 항목

기반 구현(브랜치 `feature/method-registry-foundation`)의 최종 전체 리뷰에서 확인된, 이후
단계에서 처리해야 할 것들. 기반 자체의 정확성에는 영향이 없으나 그대로 두면 나중에 문제가 된다.

- **Phase 4 서버는 `check_index_meta(..., n_vectors=index.ntotal)`을 반드시 넘겨야 한다.**
  `n_vectors`는 빌드 결과라 지문 입력이 될 수 없어 선택 인자로 뒀다. 넘기지 않으면
  잘리거나 뒤바뀐 `faiss.index`를 잡지 못한다.
- **`index_arm`은 `index_meta.json` 존재만 보고 재빌드를 건너뛰며 `check_index_meta`를
  호출하지 않는다.** `--quick` 실행 후 전체 실행을 하면 2,000장 인덱스가 전체 데이터
  평가 옆에 남는다. 서빙 시점에는 지문이 잡지만 러너는 못 잡는다.
- **`train.seed`가 지문에 없다.** `--limit`이 같아도 seed가 다르면 다른 레코드를
  인덱싱하는데 지문은 동일하게 나온다.
- **`rel_posix`는 절대경로만 정규화한다.** `./outputs/...`와 `outputs/...`는 여전히 다르게
  비교된다.
- **`train_arm`의 스킵 경로가 `write_data_pointer`를 호출하지 않는다.** 없거나 오래된
  포인터 파일이 재생성되지 않는다.
- **본 문서 §2의 산출물 레이아웃과 구현이 다르다.** 스펙은 `adapter/final/`과
  `data/pairs.jsonl`로 적었으나 구현은 `final/`과 `data/pairs.jsonl.pointer.json`을 쓴다.
  Phase 4 서버를 스펙만 보고 작성하면 어댑터를 못 찾는다. 구현 쪽 경로가 사실이다.
- **`scripts/build_subset.py`는 로카르노 코드마다 최소 1개 디자인을 항상 남긴다.**
  `--target`이 작으면(예: 2000) 271개 코드 × 약 8뷰 ≈ 2,200장이 되어 목표를 넘긴다.
  희귀 코드가 통째로 사라지는 것보다 낫다는 판단이며, 실운용 지점(10만)에서는 무시할 수준.

### 7.2 hobit 구현 후 남은 후속 항목

`feature/hobit-sampler` 최종 리뷰에서 확인된, 정확성에는 영향이 없으나 남겨둔 것들.

- **tqdm 출력량**: `run_ablation.run()`이 `text=True`로 자식을 읽어 tqdm의 ``가 개행으로
  번역된다. 갱신마다 `train.log` 한 줄 + 콘솔 한 줄이 되어 refresh 1회에 수만 줄이 될 수
  있다. `src/embed.py`의 인덱스 빌드도 같은 양상이라 기존 패턴이지만, 로그가 시끄러우면
  `mininterval=5`로 완화한다.
- **`train.py`의 `num_workers` 전달에 테스트 없음**: 조용히 직렬로 되돌아가도 아무것도
  실패하지 않고 에폭당 약 70분이 늘어난다.
- **refresh 인코딩 배치가 64로 하드코딩**: 학습 배치는 32(`configs/lora_clip.yaml`).
- **병렬 디코드 경로는 425k 규모에서 미검증**: 21 레코드 등가성 테스트로 정확성(피클링,
  행 오프셋 복원, 실패 카운트)은 고정했으나 실제 규모의 wall-clock과 호스트 RAM 피크는
  첫 실전 실행에서 확인해야 한다.
- **hobit은 다른 arm과 달리 완전 재현되지 않는다**: 배치가 GPU 임베딩에 의존하므로 같은
  seed라도 배치 순서가 달라질 수 있다. Δ의 실행 간 분산이 다른 arm보다 크다.

## 7.3 설계 단계의 미결 사항


- HOBIT 전문 미확보 (§1.3). `hobit_score` 선택자로 분리해 논문 입수 시 교체한다.
- ~~`hires384`의 huge-378 저장소 id 확인~~ **해소(2026-08-11)**: `facebook/metaclip-2-worldwide-huge-378`이
  존재한다. 같은 ViT-H 계열이므로 해상도 축만 바뀌며, 방법 구성을 바꿀 필요가 없다.
  `image_size`는 378. 아래 원문은 기록용으로 남긴다.
- (원문) `hires384`의 정확한 huge-378 HF 저장소 id는 Phase 3 착수 시 확인해 확정한다. 로컬에는
  224 모델만 있다. huge 계열 378 변형을 받을 수 없으면 이 방법은 축이 오염되므로
  (§3.4) 다른 방법으로 대체한다.
- ACCURACY.md가 FG-CLIP 2를 한국어 관련 베이스 모델 후보로 기술한 부분은 그 모델이
  영어-중국어 이중언어라는 조사 결과와 어긋난다. 본 설계 범위 밖이나 문서 수정 후보.
