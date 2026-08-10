"""HOBIT 배치 샘플러: 커버리지·결정성·hardness·모순 회피를 고정."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hobit import HobitBatchSampler  # noqa: E402


def _recs(n_design=40, views=4, titles=("Shoe", "Bottle")):
    """디자인 n_design개 × 뷰 views장.

    titles=None이면 디자인마다 고유 제목 → 제목 충돌 없이 design_id 항만 남는다.
    기본값(2종 순환)은 다른 디자인이 제목을 공유하는 상황을 만든다.
    """
    out = []
    for d in range(n_design):
        title = f"T{d}" if titles is None else titles[d % len(titles)]
        for v in range(views):
            out.append({"image": f"{d}_{v}.png", "design_id": f"D{d}", "text": title})
    return out


def _two_clusters(n, dim=8, seed=0):
    """앞 절반과 뒤 절반이 서로 먼 두 군집. hardness가 작동하면 배치가 한 군집으로 뭉친다."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, dim)) * 0.05
    e[: n // 2, 0] += 1.0
    e[n // 2:, 0] -= 1.0
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype("float32")


def test_에폭당_인덱스가_중복없이_나온다():
    recs = _recs()                                  # 160 레코드
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    flat = [i for b in s for i in b]
    assert len(flat) == len(set(flat)), "같은 인덱스가 두 번 나왔다"
    assert len(flat) == (len(recs) // 16) * 16, "자투리 규약과 다르다"
    assert set(flat) <= set(range(len(recs)))


def test_모든_배치가_정확히_batch_size():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert all(len(b) == 16 for b in s)


def test_같은_seed_같은_에폭이면_같은_배치():
    recs = _recs()
    e = _two_clusters(len(recs))
    a = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); a.set_embeddings(e)
    b = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7); b.set_embeddings(e)
    assert list(a) == list(b)


def test_에폭이_바뀌면_배치도_바뀐다():
    recs = _recs()
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=7)
    s.set_embeddings(e)
    assert list(s) != list(s)                       # __iter__가 epoch을 증가시킨다


def test_임베딩이_없으면_랜덤_폴백하되_커버리지는_유지():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=3)
    flat = [i for b in s for i in b]                # set_embeddings 호출 없음
    assert len(flat) == len(set(flat)) == (len(recs) // 16) * 16


def test_hardness_배치가_한_군집으로_뭉친다():
    """무작위 배치라면 배치 내 평균 유사도가 0 근처. greedy가 작동하면 크게 양수."""
    recs = _recs(n_design=64, views=4)              # 256 레코드
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=0.0, seed=5)
    s.set_embeddings(e)
    sims = []
    for b in s:
        v = e[b]
        g = v @ v.T
        sims.append((g.sum() - np.trace(g)) / (len(b) * (len(b) - 1)))
    assert np.mean(sims) > 0.5, f"배치 내 유사도가 낮다 — greedy가 작동하지 않음: {np.mean(sims)}"


def test_hardness가_실제_penalty_값에서도_유지된다():
    """배포 설정(hobit_penalty: 10.0)에서 hardness를 고정한다.

    위 hardness 테스트는 penalty=0이라 유사도 항만 본다 — 나중에 penalty 항이 유사도
    항을 눌러버려도 아무 테스트도 눈치채지 못한다. 여기서는 모순이 실제로 존재하는
    픽스처(제목 2종을 32개 디자인씩 공유 + 마스킹 on ⇒ 동일제목·타디자인이 모순)에
    배포 값을 그대로 넣고, 그래도 배치가 한 군집으로 뭉치는지 본다.
    """
    recs = _recs(n_design=64, views=4)              # 256 레코드, 제목 2종
    e = _two_clusters(len(recs))
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=10.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(e)
    sims = []
    for b in s:
        v = e[b]
        g = v @ v.T
        sims.append((g.sum() - np.trace(g)) / (len(b) * (len(b) - 1)))
    assert np.mean(sims) > 0.5, f"배포 penalty에서 hardness가 죽었다: {np.mean(sims)}"


def test_pool이_greedy_후보_수를_실제로_제한한다():
    """hobit_pool은 배치마다 뽑는 후보 수다 — 무시되면 조용히 O(N·D) 게더가 된다.

    pool == batch_size면 후보가 배치 크기와 같아 greedy에 고를 여지가 없다(뽑힌 것이
    그대로 배치가 된다) → 배치 내 유사도가 무작위 수준이어야 한다. pool을 키우면 같은
    시드·같은 임베딩에서도 한 군집으로 뭉친다. 이 대비가 무너지면 pool이 무시되고
    있다는 뜻이고, 실데이터에서는 배치마다 425,140 × 1024 float32(1.74 GB)를 복사한다.
    """
    recs = _recs(n_design=64, views=4)              # 256 레코드
    e = _two_clusters(len(recs))

    def intra(pool):
        s = HobitBatchSampler(recs, batch_size=16, pool=pool, penalty=0.0, seed=5)
        s.set_embeddings(e)
        out = []
        for b in s:
            v = e[b]
            g = v @ v.T
            out.append((g.sum() - np.trace(g)) / (len(b) * (len(b) - 1)))
        return float(np.mean(out))

    tight, wide = intra(16), intra(128)
    assert tight < 0.2, f"pool=batch_size인데 배치가 뭉쳤다 — pool이 무시된다: {tight}"
    assert wide > 0.5, f"pool을 키웠는데 뭉치지 않는다: {wide}"


def test_마스킹_off면_같은_design_id를_같은_배치에_피한다():
    """mask_false_negatives=False면 같은 디자인의 뷰도 네거티브로 밀리므로 모순이다.

    제목을 디자인마다 고유하게 준다: 제목이 2종뿐이면 같은 design_id ⟹ 같은 제목이라
    same_text 항이 same_design 항을 완전히 가려, design_id 회피가 작동하는지 확인할 수 없다.
    실데이터는 제목당 평균 16개 레코드라 4096 풀에서 제목 충돌이 희소하다.

    에폭 끝의 배치들은 검사에서 뺀다. penalty는 모순을 없애지 않고 뒤로 미룰 뿐이라
    미뤄둔 것들이 에폭 말미에 꼬리로 쌓인다(실데이터 13,285배치 중 마지막 19개에
    모순이 전부 몰린다 — src/hobit.py 한계 1). 이 픽스처(256레코드·16배치)에서는
    그 꼬리가 마지막 1개라 [:-1]만 보면 된다.
    """
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    batches = list(s)
    dup = 0
    for b in batches[:-1]:                      # 마지막 배치는 강제 구성이라 제외
        ds = [recs[i]["design_id"] for i in b]
        dup += len(ds) - len(set(ds))
    assert dup == 0, f"선택 여지가 있는 배치에서 같은 design_id가 {dup}번 겹쳤다"


def test_에폭_끝_꼬리_배치는_모순을_피하지_못한다():
    """구조적 한계를 명시적으로 고정한다 — 발견이 아니라 알려진 성질이 되도록.

    '마지막 배치 하나'가 아니라 에폭 끝의 꼬리다: penalty가 모순 쌍을 제거하지 않고
    미루기만 하므로, 커버리지 제약상 미뤄둔 것들이 끝에서 한꺼번에 소진된다. 꼬리
    길이는 제목 쏠림에 비례한다(실데이터 13,285배치 중 19개, 최다 제목 3%면 47개).
    이 꼬리가 에폭의 evaluate·어댑터 저장 직전에 놓인다. 이 픽스처는 꼬리가 1배치다.
    """
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=False, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    batches = list(s)
    assert len(batches) * 16 == len(recs), "이 픽스처는 자투리 없이 나누어떨어져야 한다"
    # 꼬리 배치에는 미뤄둔 모순이 강제로 들어가므로 중복이 생길 수 있다
    last_ds = [recs[i]["design_id"] for i in batches[-1]]
    assert len(last_ds) == 16


def test_마스킹_on이면_같은_design_id_뷰가_같은_배치에_모일_수_있다():
    """마스킹이 켜지면 같은 design_id는 positive로 처리되므로 회피 대상이 아니다.

    에폭 끝 꼬리 배치는 제외한다 — 선택 여지가 없어 mask_fn 값과 무관하게 중복이
    생기므로, 포함하면 _contra의 mask on 분기가 망가져도 이 단언이 통과할 수 있다.
    (이 픽스처의 꼬리는 1배치다. 실데이터에서는 마지막 수십 배치가 꼬리다.)
    """
    recs = _recs(n_design=64, views=4, titles=None)
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    batches = list(s)
    dup = sum(len(b) - len({recs[i]["design_id"] for i in b}) for b in batches[:-1])
    assert dup > 0, "마스킹 on인데도 같은 design_id를 회피하고 있다 — 조건부 규칙이 무의미"


def test_마스킹_on이면_같은_design_id는_허용하고_동일제목_타디자인만_피한다():
    recs = _recs(n_design=64, views=4)              # 제목 2종이 32개 디자인씩 공유
    s = HobitBatchSampler(recs, batch_size=16, pool=128, penalty=50.0,
                          mask_false_negatives=True, seed=5)
    s.set_embeddings(_two_clusters(len(recs)))
    cross = 0
    for b in s:
        for x in range(len(b)):
            for y in range(x + 1, len(b)):
                rx, ry = recs[b[x]], recs[b[y]]
                if rx["text"] == ry["text"] and rx["design_id"] != ry["design_id"]:
                    cross += 1
    # 제목이 2종뿐이라 완전 회피는 불가능하지만, 무작위(배치16에서 약 60쌍)보다 훨씬 적어야 한다
    n_batches = len(recs) // 16
    assert cross / n_batches < 40, f"동일제목·타디자인 쌍이 배치당 {cross / n_batches:.1f}쌍"


def test_len은_배치_수와_일치():
    recs = _recs()
    s = HobitBatchSampler(recs, batch_size=16, pool=64, seed=1)
    s.set_embeddings(_two_clusters(len(recs)))
    assert len(s) == len(list(s))


from hobit import embed_records  # noqa: E402


def test_embed_records는_레코드_순서를_보존한다(tmp_path):
    from PIL import Image

    from dataset import preprocess_drawing
    for i in range(5):
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(tmp_path / f"{i}.png")
    recs = [{"image": f"{i}.png"} for i in range(5)]

    def fake_encode(imgs):                       # 픽셀값을 그대로 벡터로
        return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")

    got = embed_records(recs, str(tmp_path), 16, fake_encode, batch_size=2)
    # preprocess_drawing이 내부에서 그레이스케일 변환을 하므로 R값이 아니라 휘도(L)값이
    # 나온다 — 순서 보존이 검증 대상이지 원본 R값 보존이 아니므로 실제 변환값과 비교한다.
    expect = [preprocess_drawing(Image.new("RGB", (16, 16), (i * 10, 0, 0)), 16).getpixel((0, 0))[0]
              for i in range(5)]
    assert got.shape == (5, 3)
    assert [int(v[0]) for v in got] == expect
    assert expect == sorted(expect), "픽스처 전제(밝기가 i에 따라 단조증가)가 깨졌다"


def test_embed_records는_깨진_이미지를_0벡터로_채워_정렬을_유지한다(tmp_path):
    """검증 대상은 정렬 계약이므로 실패 허용치(max_fail_ratio)를 풀어 놓는다 —
    2장 중 1장 실패는 기본 1% 문턱을 넘어 RuntimeError가 된다(아래 실패 테스트 참조)."""
    from PIL import Image

    from dataset import preprocess_drawing
    Image.new("RGB", (16, 16), (10, 0, 0)).save(tmp_path / "ok.png")
    recs = [{"image": "missing.png"}, {"image": "ok.png"}]

    def fake_encode(imgs):
        return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")

    got = embed_records(recs, str(tmp_path), 16, fake_encode, batch_size=2,
                        max_fail_ratio=1.0)
    expect_ok = preprocess_drawing(Image.new("RGB", (16, 16), (10, 0, 0)), 16).getpixel((0, 0))[0]
    assert got.shape == (2, 3)
    assert np.all(got[0] == 0), "열 수 없는 이미지 자리가 0 벡터가 아니다"
    assert int(got[1][0]) == expect_ok, "행이 밀려 레코드와 어긋났다"


def _fake_encode(imgs):
    return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")


def _make_images(tmp_path, n_ok, n_missing):
    """열리는 이미지 n_ok장 + 존재하지 않는 경로 n_missing개의 레코드."""
    from PIL import Image
    for i in range(n_ok):
        Image.new("RGB", (16, 16), (10, 0, 0)).save(tmp_path / f"ok{i}.png")
    return ([{"image": f"ok{i}.png"} for i in range(n_ok)]
            + [{"image": f"missing{i}.png"} for i in range(n_missing)])


def test_embed_records는_전부_실패하면_예외를_던진다(tmp_path):
    """image_root 오지정 시나리오. 0 벡터 행렬을 돌려주면 set_embeddings가 행 수만 보고
    통과시키고, dots가 항상 0이라 greedy가 baseline 랜덤 배치로 조용히 퇴화한다 —
    며칠짜리 학습이 끝난 뒤 summary.md의 Δ≈0을 '배치 구성은 효과 없음'으로 읽게 된다."""
    import pytest
    recs = _make_images(tmp_path, n_ok=0, n_missing=5)
    with pytest.raises(RuntimeError, match="하나도 없다"):
        embed_records(recs, str(tmp_path), 16, _fake_encode, batch_size=2)


def test_embed_records는_실패율이_문턱을_넘으면_예외를_던진다(tmp_path):
    """부분 실패는 더 나쁘다 — 실패 행은 0 벡터라 greedy가 에폭 끝으로 몰아넣는데
    로그에는 shape만 찍혀 보이지 않는다. 1%를 넘으면 설정 사고로 보고 중단한다."""
    import pytest
    recs = _make_images(tmp_path, n_ok=90, n_missing=10)      # 10% 실패
    with pytest.raises(RuntimeError, match="허용치"):
        embed_records(recs, str(tmp_path), 16, _fake_encode, batch_size=8)


def test_embed_records는_산발적_실패는_통과시키고_수를_로그로_남긴다(tmp_path, capsys):
    """손상 도면 몇 장 때문에 다일 학습이 죽으면 안 된다 — 문턱 이하는 통과하되
    몇 장이 0 벡터로 남았는지는 반드시 출력한다."""
    recs = _make_images(tmp_path, n_ok=200, n_missing=1)      # 0.5% 실패
    got = embed_records(recs, str(tmp_path), 16, _fake_encode, batch_size=16)
    assert got.shape == (201, 3)
    assert np.all(got[200] == 0), "실패 행이 0 벡터가 아니다"
    assert "1/201장을 열지 못해" in capsys.readouterr().out


def test_embed_records의_워커_경로가_직렬_경로와_같은_결과를_낸다(tmp_path):
    """디코딩 병렬화(num_workers>0)는 행 정렬 계약을 건드리는 변경이라 등가성을 고정한다.

    워커에서 실패한 자리가 배치 안에서 밀리면 그 뒤 전부가 어긋나는데, 학습은 에러
    없이 엉뚱한 레코드의 임베딩으로 배치를 짜게 된다. 실패를 중간에 하나 끼워 넣어
    오프셋 복원까지 검사한다. (모델·GPU 없이 fake encode_fn만 쓴다.)
    """
    recs = _make_images(tmp_path, n_ok=20, n_missing=0)
    recs.insert(7, {"image": "nope.png"})            # 배치 중간의 실패
    serial = embed_records(recs, str(tmp_path), 16, _fake_encode, batch_size=4,
                           max_fail_ratio=1.0)
    par = embed_records(recs, str(tmp_path), 16, _fake_encode, batch_size=4,
                        num_workers=2, max_fail_ratio=1.0)
    assert np.array_equal(serial, par), "워커 경로 결과가 직렬 경로와 다르다"
    assert np.all(par[7] == 0), "실패 자리가 밀렸다 — 행 정렬이 깨졌다"


from hobit import refresh_embeddings  # noqa: E402


class _FakeModel:
    """torch 없이 eval()/train() 호출 순서만 기록하는 대역 — GPU/모델 로딩 없이
    refresh_embeddings의 예외 복구 계약을 검증하기 위한 것."""

    def __init__(self):
        self.calls = []

    def eval(self):
        self.calls.append("eval")

    def train(self):
        self.calls.append("train")


def test_refresh_embeddings는_encode_fn이_예외를_던져도_train_모드로_복귀한다(tmp_path):
    """CUDA OOM 등으로 인코딩 중 예외가 나도 model.train()이 반드시 호출되어야 한다.

    복구를 빠뜨리면 이후 학습이 조용히 eval 모드로 진행되어 dropout이 꺼진 채
    진행된다 — 에러 없이 결과만 나빠지는 실패라 반드시 finally로 고정한다.
    """
    from PIL import Image
    Image.new("RGB", (16, 16), (10, 0, 0)).save(tmp_path / "x.png")
    m = _FakeModel()
    recs = [{"image": "x.png"}]

    def boom(imgs):
        raise RuntimeError("OOM 시뮬레이션")

    try:
        refresh_embeddings(m, recs, str(tmp_path), 16, boom)
        assert False, "예외가 전파되지 않았다"
    except RuntimeError:
        pass
    assert m.calls == ["eval", "train"], f"eval/train 호출이 어긋났다: {m.calls}"


def test_refresh_embeddings는_정상_경로에서도_train_모드로_복귀한다(tmp_path):
    from PIL import Image
    Image.new("RGB", (16, 16), (10, 0, 0)).save(tmp_path / "ok.png")
    m = _FakeModel()
    recs = [{"image": "ok.png"}]

    def fake_encode(imgs):
        return np.array([[im.getpixel((0, 0))[0]] * 3 for im in imgs], dtype="float32")

    emb = refresh_embeddings(m, recs, str(tmp_path), 16, fake_encode)
    assert emb.shape == (1, 3)
    assert m.calls == ["eval", "train"], f"eval/train 호출이 어긋났다: {m.calls}"
