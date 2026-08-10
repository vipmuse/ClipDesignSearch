"""run_ablation.run()의 콘솔 인코딩 처리: 크래시 없이 살아남고, 로그는 원문을 보존한다.

실제로 겪은 버그(2026-08-11 hobit 스모크): 한국어 로그 한 줄 때문에 부모 프로세스가
UnicodeEncodeError로 죽어 몇 시간짜리 학습이 날아갈 뻔했다. 회귀 재발 방지를 위해
subprocess.Popen과 sys.stdout을 대역으로 바꿔 GPU/모델 없이 이 경로만 고정한다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_ablation  # noqa: E402

KOREAN_LINE = "[hobit] epoch 0: 임베딩 (361, 1024) 갱신\n"


class _FakeProcess:
    """subprocess.Popen 대역 — 실제 프로세스를 띄우지 않고 stdout 라인만 흉내낸다."""

    def __init__(self, lines):
        self.stdout = iter(lines)
        self.returncode = 0

    def wait(self):
        pass


class _FakeStdout:
    """sys.stdout 대역 — 좁은 codec(encoding)을 강제하고, 쓰기 실패를 흉내낼 수 있다.

    write()에서 self.encoding으로 실제 strict encode를 시도한다 — 그냥 리스트에
    문자열을 담기만 하면 run()이 safe-transform을 빼먹어도 이 대역은 절대 실패하지
    않아, 정작 지키려는 회귀를 못 잡는 테스트가 된다. 진짜 콘솔(TextIOWrapper)이
    표현 못 하는 문자에 UnicodeEncodeError를 던지는 것과 같은 조건으로 맞춘다.
    """

    def __init__(self, encoding="ascii", fail_after=None):
        self.encoding = encoding
        self.written = []
        self._n = 0
        self.fail_after = fail_after

    def write(self, s):
        self._n += 1
        if self.fail_after is not None and self._n > self.fail_after:
            raise OSError("콘솔에 쓸 수 없음 (시뮬레이션)")
        s.encode(self.encoding)                   # 표현 못 하면 실제 콘솔처럼 예외
        self.written.append(s)

    def flush(self):
        pass


class _FailingPrint:
    """print() 대역 — allow번째 호출까지는 통과, 그 뒤로는 콘솔 출력 실패를 흉내낸다.

    print() 한 번이 내부적으로 sys.stdout.write()를 몇 번 호출하는지는 구현 세부라
    write() 레벨에서 카운트하면 깨지기 쉽다. print() 호출 자체를 세는 편이 안정적이다.
    """

    def __init__(self, allow=1):
        self.allow = allow
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self.allow:
            raise OSError("콘솔 출력 실패 (시뮬레이션)")


def _patch_popen(monkeypatch, lines):
    monkeypatch.setattr(run_ablation.subprocess, "Popen",
                         lambda *a, **k: _FakeProcess(lines))


def test_비ASCII_라인이_있어도_run이_예외없이_끝난다(tmp_path, monkeypatch):
    """콘솔 codec이 좁아도(ascii) run()이 UnicodeEncodeError로 죽으면 안 된다 —
    실제로 죽어서 학습이 날아간 버그다."""
    monkeypatch.setattr(sys, "stdout", _FakeStdout(encoding="ascii"))
    _patch_popen(monkeypatch, [KOREAN_LINE])
    log_path = str(tmp_path / "train.log")

    run_ablation.run(["fake", "cmd"], log_path)   # 예외가 나면 이 줄에서 테스트 실패


def test_로그파일은_비ASCII_원문을_손상없이_보존한다(tmp_path, monkeypatch):
    """콘솔 echo의 errors='replace' 손실은 화면에만 적용돼야 한다 — 로그 파일에는
    원문이 그대로 남아야 재현 기록으로서 의미가 있다."""
    monkeypatch.setattr(sys, "stdout", _FakeStdout(encoding="cp949"))
    _patch_popen(monkeypatch, [KOREAN_LINE])
    log_path = str(tmp_path / "train.log")

    run_ablation.run(["fake", "cmd"], log_path)

    saved = open(log_path, encoding="utf-8").read()
    assert KOREAN_LINE in saved, "로그 파일이 원문 대신 손상된 텍스트를 담고 있다"


def test_콘솔_출력이_실패해도_로그에는_이미_기록되어_있다(tmp_path, monkeypatch):
    """log.write가 콘솔 print보다 먼저 실행돼야 한다는 순서 보장을 고정한다.
    누군가 두 줄을 다시 뒤바꾸면 이 테스트가 잡는다."""
    monkeypatch.setattr(run_ablation, "print", _FailingPrint(allow=1), raising=False)
    _patch_popen(monkeypatch, [KOREAN_LINE])
    log_path = str(tmp_path / "train.log")

    with pytest.raises(OSError):
        run_ablation.run(["fake", "cmd"], log_path)

    saved = open(log_path, encoding="utf-8").read()
    assert KOREAN_LINE in saved, "콘솔 출력이 실패하기 전에 로그 기록이 이미 끝나 있어야 한다"
