"""웹 서버의 방법 선택·비교 로직 테스트 (모델·인덱스 없이 순수 함수만)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi import HTTPException

import server


def _payload(method, designs):
    """method의 topk 결과 스텁. designs = [(design_id, title), ...] 순위순."""
    return {"method": method, "results": [
        {"design_id": d, "patent_no": d.split("-")[0], "title": t,
         "image_url": f"/api/image/{k}", "score": 1.0 - 0.01 * k}
        for k, (d, t) in enumerate(designs)]}


def test_union은_공통_발견을_앞세우고_방법별_순위를_담는다():
    a = _payload("loracap", [("D1-x", "chair"), ("D2-x", "table"), ("D3-x", "lamp")])
    b = _payload("hires378", [("D2-x", "table"), ("D4-x", "sofa")])
    u = server._union([a, b])
    by_id = {d["design_id"]: d for d in u}
    assert by_id["D2-x"]["picks"] == {"loracap": 2, "hires378": 1}
    # 두 방법 모두 발견한 D2가 단독 발견들보다 앞선다
    assert u[0]["design_id"] == "D2-x"
    # 단독 발견끼리는 순위가 높은 쪽이 앞
    solo = [d["design_id"] for d in u[1:]]
    assert solo.index("D1-x") < solo.index("D3-x")


def test_parse_methods는_중복을_제거하고_상한을_지킨다(monkeypatch):
    monkeypatch.setattr(server, "METHODS", {n: {} for n in
                        ("loracap", "hires378", "baseline", "hobit", "tic")})
    monkeypatch.setattr(server, "STATE", {"default": "loracap"})
    assert server._parse_methods("", "") == ["loracap"]                     # 기본값
    assert server._parse_methods("hobit", "") == ["hobit"]                  # 단일
    assert server._parse_methods("", "loracap, hires378,loracap") == \
        ["loracap", "hires378"]                                             # 중복 제거
    got = server._parse_methods("", "loracap,hires378,baseline,hobit,tic")
    assert len(got) == 5, "7개 arm 전체 비교를 허용한다 (상한 8)"
    with pytest.raises(HTTPException):
        server._parse_methods("없는방법", "")


def test_method_paths_base는_baseline_설정을_빌린다():
    cfg, adapter = server._method_paths("base")
    assert "baseline" in cfg and adapter == "none"
    cfg2, adapter2 = server._method_paths("loracap")
    assert "loracap" in cfg2 and adapter2.endswith("final")
