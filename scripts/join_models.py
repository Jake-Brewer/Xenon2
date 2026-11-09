"""Joins models/<name>.bin from its LFS parts (<name>.bin.part1, .part2, ...) when it is missing.

GitHub's LFS takes files up to 2 GB, so each model is stored in parts. The desktop app joins them
itself at startup (app/src-tauri/src/model_parts.rs); this script does the same for a fresh
checkout or for command-line tools. Safe to run any time: models already joined are skipped.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

LFS_POINTER = b"version https://git-lfs"


def join(model: Path, quiet: bool) -> bool:
    parts = []
    n = 1
    while (part := model.with_name(f"{model.name}.part{n}")).exists():
        parts.append(part)
        n += 1
    if model.exists() or not parts:
        return True
    expected = int(model.with_name(model.name + ".size").read_text().strip())
    for part in parts:
        with part.open("rb") as f:
            if f.read(len(LFS_POINTER)) == LFS_POINTER:
                print(f"{part.name} is still an LFS pointer. Run `git lfs pull` first.", file=sys.stderr)
                return False
    joining = model.with_name(model.name + ".joining")
    with joining.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out, 16 * 1024 * 1024)
    if joining.stat().st_size != expected:
        joining.unlink()
        print(f"{model.name}: joined size is wrong; a part is missing or damaged.", file=sys.stderr)
        return False
    os.replace(joining, model)
    if not quiet:
        print(f"joined {model.name} from {len(parts)} parts")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print only problems")
    args = parser.parse_args()
    models = Path(__file__).resolve().parent.parent / "models"
    if not models.is_dir():
        return 0
    names = {p.name.split(".bin.part")[0] + ".bin" for p in models.glob("*.bin.part*")}
    ok = all([join(models / name, args.quiet) for name in sorted(names)])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
