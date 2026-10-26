"""连续构建两次，检查跟踪文件及 Git 状态；第二次禁止 Python 外网连接。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CASE = Path(__file__).resolve().parents[1]
REPOSITORY = CASE.parents[1]

# 单独进程的 sitecustomize 只影响本次构建及其 Python 内核；本地 ZMQ 通信保留。
OFFLINE_GUARD = """import socket
def _check(address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise OSError("Offline build: external network disabled")
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
def connect(self, address):
    _check(address)
    return _connect(self, address)
def connect_ex(self, address):
    _check(address)
    return _connect_ex(self, address)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
"""


def tracked_state() -> dict:
    """记录整个仓库的跟踪文件内容和状态，不要求当前分支没有待提交修改。"""
    paths = (
        subprocess.check_output(["git", "-C", str(REPOSITORY), "ls-files", "-z"])
        .decode("utf-8")
        .split("\0")
    )
    hashes = {}
    for relative in filter(None, paths):
        path = REPOSITORY / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    status = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "status", "--porcelain", "-uall"]
    ).decode("utf-8")
    return {"files": hashes, "status": status}


def main() -> None:
    baseline = tracked_state()
    guard = CASE / "build/offline_guard"
    guard.mkdir(parents=True, exist_ok=True)
    (guard / "sitecustomize.py").write_text(OFFLINE_GUARD, encoding="utf-8")
    runs = []
    for number in (1, 2):
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if number == 2:
            environment["PYTHONPATH"] = str(guard) + os.pathsep + environment.get("PYTHONPATH", "")
            # 在隔离子进程中先证明网络拦截生效，再启动构建和 nbclient 子内核。
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import socket; s=socket.socket(); "
                    "\ntry: s.connect(('example.invalid',443))"
                    "\nexcept OSError as e: assert 'external network disabled' in str(e)"
                    "\nelse: raise AssertionError('Network guard did not load')",
                ],
                env=environment,
                check=True,
                cwd=CASE,
            )
        subprocess.run(
            [sys.executable, "scripts/build_notebooks.py"], cwd=CASE, env=environment, check=True
        )
        subprocess.run(
            [sys.executable, "scripts/verify_site.py"], cwd=CASE, env=environment, check=True
        )
        if tracked_state() != baseline:
            raise RuntimeError(f"Build {number} changed tracked files or Git status")
        runs.append(
            {
                "build": number,
                "python_external_network_blocked": number == 2,
                "tracked_hashes_unchanged": True,
                "git_status_unchanged": True,
            }
        )
    result = {
        "runs": runs,
        "baseline_clean": not baseline["status"],
        "tracked_files": len(baseline["files"]),
    }
    (CASE / "build/reproducibility.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Two builds passed; tracked files and Git status unchanged.")


if __name__ == "__main__":
    main()
