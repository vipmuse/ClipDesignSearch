"""리트리벌 메트릭 (numpy 전용 — torch 없이 단위 테스트 가능).

eval_retrieval.py가 쿼리를 청크로 나눠 호출한다.
"""
import numpy as np


def rank_metrics(sim, rel, ks=(1, 5, 10)):
    """유사도/정답 행렬 → 쿼리별 히트·AP.

    sim: [Q, N] 유사도 (클수록 유사)
    rel: [Q, N] bool 정답 (자기 자신 제외는 호출부 책임)
    반환: (hits: {k: bool[Q]}, ap: float[Q], valid: bool[Q])
      valid: 정답이 1개 이상 있는 쿼리만 True — 집계 시 이것으로 필터.
    """
    order = np.argsort(-sim, axis=1, kind="stable")
    rs = np.take_along_axis(rel, order, axis=1)          # 정렬 순서의 정답 여부
    hits = {k: rs[:, :k].any(axis=1) for k in ks}
    cum = np.cumsum(rs, axis=1)
    prec = cum / np.arange(1, rs.shape[1] + 1)[None, :]  # precision@rank
    n_rel = rs.sum(axis=1)
    valid = n_rel > 0
    ap = (prec * rs).sum(axis=1) / np.maximum(n_rel, 1)  # average precision
    return hits, ap, valid


class MetricAccumulator:
    """청크 단위 rank_metrics 결과를 모아 최종 R@K / mAP 산출."""

    def __init__(self, ks=(1, 5, 10)):
        self.ks = ks
        self.hits = {k: 0 for k in ks}
        self.ap_sum = 0.0
        self.n = 0

    def update(self, hits, ap, valid):
        for k in self.ks:
            self.hits[k] += int(hits[k][valid].sum())
        self.ap_sum += float(ap[valid].sum())
        self.n += int(valid.sum())

    def result(self):
        if self.n == 0:
            return {}
        out = {f"R@{k}": self.hits[k] / self.n for k in self.ks}
        out["mAP"] = self.ap_sum / self.n
        out["queries"] = self.n
        return out
