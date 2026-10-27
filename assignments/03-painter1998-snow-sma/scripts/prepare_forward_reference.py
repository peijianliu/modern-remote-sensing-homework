"""显式生成或登记完整 MATLAB 参考；默认教学不调用此入口。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from scipy.io import loadmat

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from scripts.run_supplement import resolve_matlab


def prepare_reference(
    output: Path,
    source: Path | None = None,
    workspace: Path | None = None,
    matlab: str | None = None,
) -> None:
    """保存完整参考到忽略目录，保留已有文件。

    source 提供已经运行的 MAT；否则 workspace 必须包含官方光学库、SNICAR
    工作副本及配置，再用显式指定或已配置的 MATLAB 运行 run_forward。
    """
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Reference output exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        data = loadmat(source, variable_names=["ww_albedo", "snicar_albedo"])
        if any(
            key not in data or data[key].shape != (148, 480, 3, 2)
            for key in ("ww_albedo", "snicar_albedo")
        ):
            raise ValueError("Reference must contain complete 148 x 480 x 3 x 2 arrays")
        shutil.copyfile(source, output)
    else:
        if workspace is None:
            raise ValueError("Provide --source-mat or a prepared --workspace")
        workspace = workspace.resolve()
        for relative in (
            "configs/baseline.json",
            "data/sources/optics_extraction.json",
            "data/cache/matlab/snicarAD_v3.m",
        ):
            if not (workspace / relative).is_file():
                raise FileNotFoundError(f"Prepared reference workspace is missing {relative}")

        def quote(path: Path) -> str:
            return str(path).replace("\\", "/").replace("'", "''")

        expression = (
            f"addpath('{quote(CASE / 'matlab')}');"
            f"addpath('{quote(CASE / 'vendor/SPIRES')}');"
            f"run_forward('{quote(output)}',[0 60 22]);"
        )
        subprocess.run(
            [str(resolve_matlab(matlab)), "-batch", expression], cwd=workspace, check=True
        )
    print(f"Reference SHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-reference", action="store_true", required=True)
    parser.add_argument("--source-mat", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--matlab")
    parser.add_argument("--output", type=Path, default=CASE / "data/cache/forward_models.mat")
    args = parser.parse_args()
    if args.source_mat is not None and (args.workspace is not None or args.matlab is not None):
        parser.error("Use either an existing MAT or MATLAB with a prepared workspace")
    prepare_reference(args.output, args.source_mat, args.workspace, args.matlab)


if __name__ == "__main__":
    main()
