"""Unit tests for training/idle_train.py gate functions."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import idle_train as it


class TestGpuLeaseFree:
    def test_no_lease_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(it, "LEASE_FILE", tmp_path / "LEASE.txt")
        ok, reason = it.gpu_lease_free()
        assert ok
        assert "no lease file" in reason

    def test_lease_with_game_signal(self, tmp_path, monkeypatch):
        lease = tmp_path / "LEASE.txt"
        lease.write_text("steam is using the GPU\n")
        monkeypatch.setattr(it, "LEASE_FILE", lease)
        ok, reason = it.gpu_lease_free()
        assert not ok
        assert "steam" in reason

    def test_lease_with_no_game(self, tmp_path, monkeypatch):
        lease = tmp_path / "LEASE.txt"
        lease.write_text("xenon2-inference is running\n")
        monkeypatch.setattr(it, "LEASE_FILE", lease)
        ok, reason = it.gpu_lease_free()
        assert ok

    def test_codingame_blocked(self, tmp_path, monkeypatch):
        lease = tmp_path / "LEASE.txt"
        lease.write_text("codingame-e2 contest running\n")
        monkeypatch.setattr(it, "LEASE_FILE", lease)
        ok, _ = it.gpu_lease_free()
        assert not ok


class TestGpuTemp:
    def test_returns_none_when_nvidia_smi_missing(self, monkeypatch):
        def fake_run(*a, **kw):
            raise FileNotFoundError
        monkeypatch.setattr(it.subprocess, "run", fake_run)
        assert it.gpu_temp() is None

    def test_parses_temperature(self, monkeypatch):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "68\n"
        monkeypatch.setattr(it.subprocess, "run", lambda *a, **kw: result)
        assert it.gpu_temp() == 68


class TestGpuCoolEnough:
    def test_no_nvidia_smi_is_ok(self, monkeypatch):
        monkeypatch.setattr(it, "gpu_temp", lambda: None)
        ok, reason = it.gpu_cool_enough()
        assert ok
        assert "CPU" in reason

    def test_cool_gpu_passes(self, monkeypatch):
        monkeypatch.setattr(it, "gpu_temp", lambda: 65)
        ok, _ = it.gpu_cool_enough()
        assert ok

    def test_hot_gpu_fails(self, monkeypatch):
        monkeypatch.setattr(it, "gpu_temp", lambda: 75)
        ok, reason = it.gpu_cool_enough()
        assert not ok
        assert "75" in reason

    def test_exactly_cool_c_passes(self, monkeypatch):
        monkeypatch.setattr(it, "gpu_temp", lambda: it.COOL_C)
        ok, _ = it.gpu_cool_enough()
        assert ok
