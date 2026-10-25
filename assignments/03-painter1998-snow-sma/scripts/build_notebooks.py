"""执行三份教学 Notebook，并导出带本地公式资源的 HTML。

运行目录不影响数据定位。实际执行使用当前 Python 的临时 kernel spec，
防止系统已注册的 python3 内核指向另一套环境。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

CASE = Path(__file__).resolve().parents[1]
REPOSITORY = CASE.parents[1]
NOTEBOOK_NAMES = ("01_snow_optics", "02_fig3_sma", "03_aviris_modis")
MATHJAX_VERSION = "3.2.2"


def prepare_mathjax(asset_directory: Path) -> None:
    """从官方 npm registry 获取固定版本，校验 SHA-512 后仅提取完整 SVG 组件。"""
    asset_directory.mkdir(parents=True, exist_ok=True)
    registry = f"https://registry.npmjs.org/mathjax/{MATHJAX_VERSION}"
    with urlopen(registry, timeout=60) as response:
        metadata = json.load(response)
    url = metadata["dist"]["tarball"]
    if urlparse(url).hostname != "registry.npmjs.org":
        raise ValueError("Unexpected MathJax package host")
    with urlopen(url, timeout=120) as response:
        payload = response.read(40_000_001)
    if len(payload) > 40_000_000:
        raise ValueError("MathJax package exceeds the expected size")
    checksum = "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode("ascii")
    if checksum != metadata["dist"]["integrity"]:
        raise ValueError("MathJax package integrity mismatch")
    # tex-svg-full 包含 TeX 扩展与 SVG 字形；本案例不需要额外网络组件。
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        for member_name, output_name in (
            ("package/es5/tex-svg-full.js", "tex-svg-full.js"),
            ("package/LICENSE", "LICENSE"),
        ):
            member = archive.extractfile(member_name)
            if member is None:
                raise ValueError(f"Missing upstream package file: {member_name}")
            (asset_directory / output_name).write_bytes(member.read())
    source = {
        "package": "mathjax",
        "version": MATHJAX_VERSION,
        "url": url,
        "npm_integrity": checksum,
        "license": "Apache-2.0",
        "files": {
            name: hashlib.sha256((asset_directory / name).read_bytes()).hexdigest()
            for name in ("tex-svg-full.js", "LICENSE")
        },
    }
    (asset_directory / "SOURCE.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def offline_html(body: str, notebook_name: str | None = None) -> str:
    """移除导出模板的联网脚本，使用本地 MathJax SVG 完整组件。"""
    # 教学输出为静态图表，不需要 RequireJS、widgets 或 Mermaid 的 CDN 脚本。
    body = re.sub(r"<script\b[^>]*>.*?</script\s*>", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<link\b[^>]*href=[\"'](?:https?:)?//[^>]*>", "", body, flags=re.IGNORECASE)
    configuration = r"""<script>
window.MathJax = {
  tex: {inlineMath: [['\\(','\\)'], ['$','$']], displayMath: [['\\[','\\]'], ['$$','$$']]},
  svg: {fontCache: 'local'}, options: {enableMenu: false}
};
</script>
<script defer src="assets/mathjax/tex-svg-full.js"></script>
<style>
body {max-width: 1180px; margin: 0 auto; padding: 20px; background: #fff; color: #20242a;}
.case-navigation {font: 14px system-ui,sans-serif; padding: 10px 0 18px; border-bottom: 1px solid #ddd; margin-bottom: 20px;}
.case-navigation a {color: #15577b; margin-right: 20px;}
img {max-width: 100%; height: auto;}
table {border-collapse: collapse;} th,td {padding: .4em .65em;}
</style>"""
    body = body.replace("</head>", configuration + "\n</head>")
    navigation = (
        '<nav class="case-navigation"><a href="../../index.html">课程案例首页</a>'
        '<a href="index.html">Painter 1998 案例目录</a>'
    )
    if notebook_name:
        navigation += f'<a href="notebooks/{notebook_name}.ipynb">下载 Notebook</a>'
    navigation += "</nav>"
    for name in NOTEBOOK_NAMES:
        body = body.replace(f'href="{name}.ipynb"', f'href="{name}.html"')
    body = re.sub(r"(<body\b[^>]*>)", r"\1" + navigation, body, count=1, flags=re.IGNORECASE)
    if re.search(r"<(?:script|link)\b[^>]*(?:src|href)=[\"'](?:https?:)?//", body, re.I):
        raise ValueError("Remote rendering dependency remains in exported HTML")
    return body


def execute_notebook(path: Path, build_directory: Path, timeout: int) -> tuple:
    """从清空输出的 Notebook 开始执行，并关闭本次启动的内核。"""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop("execution", None)
    kernels = build_directory / "kernels"
    kernel = kernels / "case-runtime"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Case build runtime",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    manager = KernelManager(
        kernel_name="case-runtime",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
    )
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(CASE)}},
    )
    started = time.perf_counter()
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    elapsed = time.perf_counter() - started
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
        if any(output.output_type == "error" for output in cell.get("outputs", [])):
            raise ValueError(f"Notebook contains an error output: {path.name}")
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook, elapsed


def build(
    output: Path, selected: list[str], timeout: int = 300, download_assets: bool = False
) -> dict:
    """执行指定 Notebook，将实际结果写回 ipynb 并导出可离线阅读的 HTML。"""
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets/mathjax"
    if download_assets:
        prepare_mathjax(assets)
    if not all((assets / name).is_file() for name in ("tex-svg-full.js", "LICENSE", "SOURCE.json")):
        raise FileNotFoundError("Missing local MathJax assets; run once with --prepare-assets")
    build_directory = CASE / "build/notebooks"
    build_directory.mkdir(parents=True, exist_ok=True)
    exporter = HTMLExporter(template_name="lab", embed_images=True)
    exporter.exclude_input_prompt = False
    exporter.exclude_output_prompt = False
    exporter.mathjax_url = "assets/mathjax/tex-svg-full.js"
    records = []
    for name in selected:
        path = CASE / "notebooks" / (name + ".ipynb")
        notebook, elapsed = execute_notebook(path, build_directory, timeout)
        nbformat.write(notebook, path)
        download_directory = output / "notebooks"
        download_directory.mkdir(exist_ok=True)
        shutil.copy2(path, download_directory / path.name)
        body, _ = exporter.from_notebook_node(notebook)
        html = offline_html(body, name)
        html_path = output / (name + ".html")
        html_path.write_text(html, encoding="utf-8", newline="\n")
        record = {
            "notebook": path.name,
            "execution_seconds": round(elapsed, 3),
            "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
            "output_cells": sum(bool(cell.get("outputs")) for cell in notebook.cells),
            "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
        }
        records.append(record)
        print(f"BUILT {path.name}: {elapsed:.2f} s", flush=True)
    result = {
        "notebooks": records,
        "python_version": sys.version.split()[0],
        "all_outputs_executed": True,
        "rendering_resources": "local MathJax SVG",
    }
    (build_directory / "build_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY / "docs/assignments/03-painter1998"
    )
    parser.add_argument(
        "--notebook", nargs="+", choices=NOTEBOOK_NAMES, default=list(NOTEBOOK_NAMES)
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Download the fixed public MathJax package and verify npm integrity",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    build(args.output.resolve(), args.notebook, args.timeout, args.prepare_assets)


if __name__ == "__main__":
    main()
