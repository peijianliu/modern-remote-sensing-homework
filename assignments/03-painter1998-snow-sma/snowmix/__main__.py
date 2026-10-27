"""光谱混合分析实验的命令行入口。"""

import argparse
import json
import contextlib
import datetime as dt
import sys
import traceback
from pathlib import Path
from .artifacts import ROOT


def verification_arguments(config: dict) -> list[str]:
    """用当前配置定位待验证批次，避免报告入口漏传必需的实验目录。"""
    batch = ROOT / "experiments" / config["project"] / config["run_date"] / config["version"]
    return [
        sys.executable,
        str(ROOT / "scripts/verify_artifacts.py"),
        "--experiments-dir",
        str(batch),
    ]


def execute():
    """读取配置并分发实验阶段。"""
    parser = argparse.ArgumentParser(description="Painter 1998 / course snow-mixture experiments")
    parser.add_argument(
        "stage",
        choices=[
            "physics",
            "paper",
            "stability",
            "image",
            "report",
            "all",
            "supplement",
            "review-report",
        ],
    )
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.json"))
    parser.add_argument("--matlab", help="MATLAB executable path or command for supplement caches")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from .experiments import run_physics, run_paper, run_stability

    functions = {"physics": run_physics, "paper": run_paper, "stability": run_stability}
    if args.stage in {"supplement", "review-report"}:
        from scripts.run_supplement import run

        run(report_only=args.stage == "review-report", config=config, matlab_executable=args.matlab)
    elif args.stage in functions:
        functions[args.stage](config)
    elif args.stage in {"image", "report"}:
        if args.stage == "image":
            from .image import run_image

            run_image(config)
        else:
            from .report import build_report
            import subprocess
            import sys

            subprocess.run(verification_arguments(config), cwd=ROOT, check=True)
            build_report(config)
    else:
        for function in functions.values():
            function(config)
        angle60 = json.loads(json.dumps(config))
        angle60["synthetic"]["angle_deg_assumed"] = 60
        run_paper(angle60)
        from .image import run_image

        run_image(config)
        if Path(args.config).resolve() == (ROOT / "configs/baseline.json").resolve():
            alternative = json.loads((ROOT / "configs/clear_roi.json").read_text(encoding="utf-8"))
            run_image(alternative)
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, str(ROOT / "scripts/qa_sensitivity.py")], cwd=ROOT, check=True
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/resolution_control.py")], cwd=ROOT, check=True
            )
        from .report import build_report
        import subprocess
        import sys

        subprocess.run(verification_arguments(config), cwd=ROOT, check=True)
        build_report(config)


class Tee:
    """将命令输出同时写入控制台和日志。"""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()


def main():
    """配置 UTF-8 输出并保存单次命令日志。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logs = ROOT / "experiments/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (logs / f"command_{stamp}.log").open("x", encoding="utf-8") as log:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
            contextlib.redirect_stderr(Tee(sys.stderr, log)),
        ):
            try:
                execute()
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
