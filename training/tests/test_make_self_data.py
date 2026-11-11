"""Tests for the pure-logic parts of make_self_data.py and eval_repo_map.py."""

import hashlib
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add training/ to path without importing the module at top level (it calls
# sys.path.insert at import time, which is fine here too).
TRAINING = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING))


def import_make_self_data():
    """Import make_self_data without triggering the app connection (eval_repo_map.py
    tries to read code-api.json at import via map_entries — patch that away)."""
    import importlib
    # Provide a stub eval_repo_map so make_self_data can import without the app.
    stub = types.ModuleType("eval_repo_map")
    stub.ROOT = TRAINING.parent
    stub.map_entries = lambda: []
    stub.post = None
    stub.question = lambda rel: f"question for {rel}"
    sys.modules.setdefault("eval_repo_map", stub)
    return importlib.import_module("make_self_data")


msd = import_make_self_data()


class TestHeldOut:
    def test_deterministic(self):
        assert msd.held_out("hello") == msd.held_out("hello")

    def test_roughly_one_in_five(self):
        words = [f"item_{i}" for i in range(1000)]
        held = sum(1 for w in words if msd.held_out(w))
        assert 150 < held < 250, f"Expected ~200 held-out of 1000, got {held}"

    def test_specific_known_value(self):
        text = "test-anchor"
        expected = int(hashlib.sha1(text.encode()).hexdigest(), 16) % 5 == 0
        assert msd.held_out(text) == expected

    def test_empty_string(self):
        result = msd.held_out("")
        assert isinstance(result, bool)


class TestDefinitionsRegex:
    """The DEFINITIONS dict maps file extensions to regexes that extract symbol names."""

    def _match(self, ext, line):
        m = msd.DEFINITIONS[ext].match(line)
        return m.group(1) if m else None

    def test_rs_fn(self):
        assert self._match(".rs", "pub fn do_the_thing(x: i32) -> i32 {") == "do_the_thing"

    def test_rs_struct(self):
        assert self._match(".rs", "pub struct FolderCtx {") == "FolderCtx"

    def test_rs_enum(self):
        assert self._match(".rs", "pub enum Attempt {") == "Attempt"

    def test_rs_const(self):
        assert self._match(".rs", "pub const MAX_EDITS: usize = 5;") == "MAX_EDITS"

    def test_rs_private_fn(self):
        assert self._match(".rs", "fn helper(s: &str) -> bool {") == "helper"

    def test_rs_too_short(self):
        # Names < 4 chars are excluded by {3,} in the pattern.
        assert self._match(".rs", "fn foo(x: i32) {}") is None

    def test_rs_no_match_on_let(self):
        assert self._match(".rs", "    let x = 5;") is None

    def test_ts_function(self):
        assert self._match(".ts", "export function paginate(items: T[]) {") == "paginate"

    def test_ts_const_arrow(self):
        assert self._match(".ts", "export const pageCount = (n: number) => {") == "pageCount"

    def test_ts_interface(self):
        assert self._match(".ts", "export interface Config {") == "Config"

    def test_ts_class(self):
        assert self._match(".ts", "class FolderAgent {") == "FolderAgent"

    def test_py_def(self):
        assert self._match(".py", "def held_out(text):") == "held_out"

    def test_py_class(self):
        assert self._match(".py", "class MyClass:") == "MyClass"

    def test_py_too_short(self):
        assert self._match(".py", "def foo():") is None

    def test_py_indented_not_matched(self):
        # Indented defs (methods) are not at column 0, so they don't match ^
        assert self._match(".py", "    def helper(self):") is None


class TestWhereItems:
    """where_items() reads source files and yields {kind, question, answer} dicts.

    We test the logic by pointing it at a temporary directory tree.
    """

    def _run_with_tmp(self, files, tmp_path):
        """Write files to tmp_path, monkeypatch source_files, collect where_items output."""
        for rel, content in files.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        written_paths = [tmp_path / rel for rel in files]

        with patch.object(msd, "source_files", return_value=iter(written_paths)):
            with patch.object(msd, "ROOT", tmp_path):
                return list(msd.where_items())

    def test_single_definition_yields_item(self, tmp_path):
        items = self._run_with_tmp(
            {"app/src-tauri/src/lib.rs": "pub fn do_work(x: i32) {}\n"},
            tmp_path,
        )
        assert len(items) == 1
        assert items[0]["kind"] == "where"
        assert "do_work" in items[0]["question"]
        assert items[0]["answer"].endswith("lib.rs")

    def test_name_in_two_files_excluded(self, tmp_path):
        items = self._run_with_tmp(
            {
                "a/lib.rs": "pub fn shared_name() {}\n",
                "b/lib.rs": "pub fn shared_name() {}\n",
            },
            tmp_path,
        )
        names = [i["question"] for i in items]
        assert not any("shared_name" in n for n in names)

    def test_reserved_names_excluded(self, tmp_path):
        items = self._run_with_tmp(
            {"app/main.rs": "pub fn main() {}\n"},
            tmp_path,
        )
        assert not any("main" in i["question"] for i in items)

    def test_multiple_distinct_definitions(self, tmp_path):
        items = self._run_with_tmp(
            {"app/src-tauri/src/agent.rs": textwrap.dedent("""\
                pub fn plan_goal(ctx: &Ctx) {}
                pub fn locate_lines(ctx: &Ctx) {}
            """)},
            tmp_path,
        )
        assert len(items) == 2
        names = {i["question"].split("`")[1] for i in items}
        assert names == {"plan_goal", "locate_lines"}


class TestCommitItems:
    """commit_items() parses git log output. Test the parsing logic by mocking subprocess."""

    def _run(self, log_output, show_output="some diff\n"):
        import subprocess
        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if "log" in cmd:
                result.stdout = log_output
            else:
                result.stdout = show_output
            return result

        with patch.object(msd, "subprocess") as mock_sub:
            mock_sub.run = fake_run
            return list(msd.commit_items())

    def test_basic_commit(self):
        log = "abc123\x00Add feature\n\x01"
        items = self._run(log)
        assert len(items) == 1
        assert items[0]["kind"] == "commit"
        assert items[0]["answer"] == "Add feature"

    def test_coauthored_by_stripped(self):
        log = "abc123\x00Fix the bug\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n\x01"
        items = self._run(log)
        assert len(items) == 1
        assert "Co-Authored-By" not in items[0]["answer"]
        assert "Fix the bug" in items[0]["answer"]

    def test_empty_diff_skipped(self):
        log = "abc123\x00Some message\n\x01"
        items = self._run(log, show_output="   \n")
        assert items == []

    def test_empty_message_skipped(self):
        log = "abc123\x00\n\x01"
        items = self._run(log, show_output="real diff\n")
        assert items == []

    def test_diff_truncated_at_max(self):
        long_diff = "x" * (msd.MAX_DIFF_CHARS + 100)
        log = "abc123\x00Truncation test\n\x01"
        items = self._run(log, show_output=long_diff)
        assert len(items) == 1
        assert len(items[0]["question"]) < len(long_diff) + 100
        assert "diff cut" in items[0]["question"]

    def test_diff_not_truncated_when_short(self):
        short_diff = "tiny\n"
        log = "abc123\x00Short diff\n\x01"
        items = self._run(log, show_output=short_diff)
        assert "diff cut" not in items[0]["question"]

    def test_multiple_commits(self):
        log = "aaa\x00First commit\n\x01bbb\x00Second commit\n\x01"
        items = self._run(log)
        assert len(items) == 2
        answers = {i["answer"] for i in items}
        assert answers == {"First commit", "Second commit"}

    def test_entry_without_null_byte_skipped(self):
        log = "no-null-here\x01abc\x00Valid message\n\x01"
        items = self._run(log)
        assert len(items) == 1
        assert items[0]["answer"] == "Valid message"


class TestScoreNearMatch:
    """Test the near-match logic in score(): right file name but wrong path."""

    def _near(self, answer, reply):
        """Replicate the near-match check from score()."""
        import re
        ok = answer in reply
        near = not ok and re.search(
            r"\b" + re.escape(answer.rsplit("/", 1)[-1]) + r"\b", reply
        )
        return bool(near)

    def test_exact_match_not_near(self):
        assert not self._near("app/src/main.rs", "app/src/main.rs")

    def test_filename_only_is_near(self):
        assert self._near("app/src-tauri/src/inference.rs", "inference.rs")

    def test_wrong_file_not_near(self):
        assert not self._near("app/src/main.rs", "app/src/lib.rs")

    def test_partial_path_is_near(self):
        assert self._near("app/src-tauri/src/inference.rs", "src/inference.rs")

    def test_empty_reply_not_near(self):
        assert not self._near("app/src/main.rs", "")


class TestBuildSplitAssignment:
    """build() assigns split = 'eval' if held_out else 'train'."""

    def test_split_assigned_based_on_held_out(self):
        # Build a small set of fake items and verify split is assigned correctly.
        fake_items = [{"kind": "where", "question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
        with patch.object(msd, "where_items", return_value=iter(fake_items)), \
             patch.object(msd, "commit_items", return_value=iter([])), \
             patch.object(msd, "file_items", return_value=iter([])), \
             patch.object(msd, "OUT") as mock_out:
            # Make OUT.open() return a no-op context manager
            mock_out.open.return_value.__enter__ = lambda s: MagicMock()
            mock_out.open.return_value.__exit__ = MagicMock(return_value=False)
            mock_out.parent.mkdir = MagicMock()
            msd.build()

        for item in fake_items:
            expected = "eval" if msd.held_out(item["question"]) else "train"
            assert item["split"] == expected, f"item {item['question']!r}: expected {expected}, got {item['split']}"


class TestMapEntriesParsing:
    """map_entries() from eval_repo_map.py parses .xenon/repo-map.md.

    We test the parsing logic directly by calling the function with a tmp tree.
    """

    def _map_entries_with(self, tmp_path, md_text):
        """Write md_text to a temp repo-map.md, create any files referenced, and
        call map_entries with ROOT/MAP patched to the tmp tree."""
        import importlib, types

        # Remove the stub so we can import the real module.
        real_name = "eval_repo_map"
        sys.modules.pop(real_name, None)
        real = importlib.import_module(real_name)
        # Restore the stub afterwards so other tests still work.
        sys.modules[real_name] = sys.modules.get("_eval_repo_map_stub") or types.ModuleType(real_name)

        md = tmp_path / ".xenon" / "repo-map.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(md_text, encoding="utf-8")

        with patch.object(real, "ROOT", tmp_path), patch.object(real, "MAP", md):
            return list(real.map_entries())

    def test_simple_entry(self, tmp_path):
        fake = tmp_path / "app" / "main.rs"
        fake.parent.mkdir(parents=True)
        fake.write_text("fn main() {}")
        entries = self._map_entries_with(
            tmp_path, "## app/\n- app/main.rs: Entry point for the Tauri app.\n"
        )
        assert len(entries) == 1
        assert entries[0] == ("app/main.rs", "Entry point for the Tauri app.")

    def test_missing_file_skipped(self, tmp_path):
        entries = self._map_entries_with(
            tmp_path, "- nonexistent.rs: Some description.\n"
        )
        assert entries == []
