"""Unit tests for training/idle_train.py's gates."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import idle_train as it

LEASE_HEADER = (
    "# RTX 2060 lease file, shared by Claude sessions on this machine (codingame-e2, xenon2-45).\n"
    "# Before a GPU job, append: <session> | start | expected end | memory cap | shared/exclusive\n"
)


def test_the_header_alone_holds_nothing():
    # The header names codingame-e2; only job lines hold the GPU.
    assert it.lease_holders(LEASE_HEADER) == []


def test_each_job_line_is_a_holder():
    text = LEASE_HEADER + "xenon2-45 | 09:38 | while the app is open | 3 GB | shared\n\n"
    assert it.lease_holders(text) == ["xenon2-45 | 09:38 | while the app is open | 3 GB | shared"]


def test_an_indented_note_is_still_a_note():
    assert it.lease_holders("   # old job, ended\n") == []


def _smi(monkeypatch, stdout, returncode=0):
    result = MagicMock(returncode=returncode, stdout=stdout)
    monkeypatch.setattr(it.subprocess, "run", lambda *a, **kw: result)


def test_reading_parses_temperature_and_memory(monkeypatch):
    _smi(monkeypatch, "68, 412\n")
    assert it.gpu_reading() == (68, 412)


def test_no_nvidia_smi_is_no_reading(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError
    monkeypatch.setattr(it.subprocess, "run", missing)
    assert it.gpu_reading() is None


def test_garbled_output_is_no_reading(monkeypatch):
    _smi(monkeypatch, "[N/A]\n")
    assert it.gpu_reading() is None


def test_gpu_gate():
    assert it.gpu_problem((it.COOL_C, 300)) is None, "exactly cool and nearly empty"
    assert "75 C" in it.gpu_problem((75, 300))
    assert "MiB" in it.gpu_problem((60, it.BUSY_MIB + 1)), "a game or loaded model"
    assert "no NVIDIA GPU" in it.gpu_problem(None)


def test_batch_size_counts_items(tmp_path):
    batch = tmp_path / "batch.jsonl"
    assert it.batch_size(batch) == 0
    batch.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")
    assert it.batch_size(batch) == 2


def test_command_reads_the_batch_through_mnt():
    cmd = it.command(Path("C:/Users/jake/AppData/Roaming/com.xenon2.app/training/batch.jsonl"))
    assert "--data_file /mnt/c/Users/jake/AppData/Roaming/com.xenon2.app/training/batch.jsonl" in cmd
    assert "--peft state" in cmd and "--precision fp16" in cmd


def test_problems_lists_every_gate(monkeypatch, tmp_path):
    lease = tmp_path / "LEASE.txt"
    lease.write_text(LEASE_HEADER + "codingame-e2 | now | 1h | 2 GB | shared\n", encoding="utf-8")
    monkeypatch.setattr(it, "LEASE_FILE", lease)
    monkeypatch.setattr(it, "BATCH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(it, "app_running", lambda: True)
    monkeypatch.setattr(it, "gpu_reading", lambda: (79, 3000))
    found = it.problems(20)
    assert len(found) == 4, found


def test_all_clear_means_no_problems(monkeypatch, tmp_path):
    lease = tmp_path / "LEASE.txt"
    lease.write_text(LEASE_HEADER, encoding="utf-8")
    batch = tmp_path / "batch.jsonl"
    batch.write_text("{}\n" * 20, encoding="utf-8")
    monkeypatch.setattr(it, "LEASE_FILE", lease)
    monkeypatch.setattr(it, "BATCH", batch)
    monkeypatch.setattr(it, "app_running", lambda: False)
    monkeypatch.setattr(it, "gpu_reading", lambda: (60, 300))
    assert it.problems(20) == []
