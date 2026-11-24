"""Reject local experiments or build directories accidentally included in wheels."""

from pathlib import Path
import sys
from zipfile import ZipFile


def check_wheel(path):
    with ZipFile(path) as wheel:
        names = wheel.namelist()
    roots = {name.split("/", 1)[0] for name in names}
    unexpected = sorted(
        root for root in roots
        if root != "chromdyn"
        and not (root.startswith("chromdyn-") and root.endswith(".dist-info"))
    )
    if "chromdyn/__init__.py" not in names:
        raise ValueError(f"{path}: missing chromdyn/__init__.py")
    if unexpected:
        raise ValueError(
            f"{path}: unexpected wheel roots {unexpected}; "
            "limit setuptools package discovery to chromdyn and its subpackages"
        )
    print(f"Validated wheel contents: {Path(path).name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/check_wheel_contents.py path/to/package.whl")
    for wheel_path in sys.argv[1:]:
        check_wheel(wheel_path)
