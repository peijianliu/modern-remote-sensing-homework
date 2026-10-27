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
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import nbformat
from bs4 import BeautifulSoup
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

CASE = Path(__file__).resolve().parents[1]
REPOSITORY = CASE.parents[1]
SITE_SOURCE = REPOSITORY / "site"
CASE_URL = Path("assignments/03-painter1998")
NOTEBOOK_NAMES = ("01_snow_optics", "02_fig3_sma", "03_aviris_modis")
NOTEBOOK_TITLES = ("01 雪光学", "02 Fig. 3 与解混", "03 实景与尺度")
MATHJAX_VERSION = "3.2.2"
MATHJAX_SOURCE = SITE_SOURCE / CASE_URL / "assets/mathjax/SOURCE.json"
MATHJAX_CACHE = REPOSITORY / "build/assets/mathjax"


def prepare_mathjax(asset_directory: Path) -> None:
    """按版本化来源锁下载组件，校验固定 SHA-512/SHA-256 后写入忽略目录。"""
    asset_directory.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(MATHJAX_SOURCE.read_text(encoding="utf-8"))
    if metadata["version"] != MATHJAX_VERSION:
        raise ValueError("MathJax version differs from the fixed source lock")
    url = metadata["url"]
    if urlparse(url).hostname != "registry.npmjs.org":
        raise ValueError("Unexpected MathJax package host")
    with urlopen(url, timeout=120) as response:
        payload = response.read(40_000_001)
    if len(payload) > 40_000_000:
        raise ValueError("MathJax package exceeds the expected size")
    checksum = "sha512-" + base64.b64encode(hashlib.sha512(payload).digest()).decode("ascii")
    if checksum != metadata["npm_integrity"]:
        raise ValueError("MathJax package integrity mismatch")
    # tex-svg-full 包含 TeX 扩展与 SVG 字形；本案例不需要额外网络组件。
    prepared = {}
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        for member_name, output_name in (
            ("package/es5/tex-svg-full.js", "tex-svg-full.js"),
            ("package/LICENSE", "LICENSE"),
        ):
            member = archive.extractfile(member_name)
            if member is None:
                raise ValueError(f"Missing upstream package file: {member_name}")
            content = member.read()
            if hashlib.sha256(content).hexdigest() != metadata["files"][output_name]:
                raise ValueError(f"MathJax extracted checksum mismatch: {output_name}")
            prepared[output_name] = content
    for name, content in prepared.items():
        (asset_directory / name).write_bytes(content)
    (asset_directory / "SOURCE.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def offline_html(
    body: str, notebook_name: str | None = None, image_descriptions: list | None = None
) -> str:
    """移除导出模板的联网脚本，使用本地 MathJax SVG 完整组件。"""
    # 教学输出为静态图表，不需要 RequireJS、widgets 或 Mermaid 的 CDN 脚本。
    body = re.sub(r"<script\b[^>]*>.*?</script\s*>", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<link\b[^>]*href=[\"'](?:https?:)?//[^>]*>", "", body, flags=re.IGNORECASE)
    configuration = r"""<script>
window.MathJax = {
  tex: {inlineMath: [['\\(','\\)'], ['$','$']], displayMath: [['\\[','\\]'], ['$$','$$']]},
  svg: {fontCache: 'local'}, options: {enableMenu: false}
};
window.addEventListener('DOMContentLoaded', () => {
  if (window.matchMedia('(max-width: 600px)').matches) {
    document.querySelectorAll('.case-toc').forEach(item => item.open = false);
  }
});
</script>
<script defer src="assets/mathjax/tex-svg-full.js"></script>
<style>
body {max-width: 1180px; margin: 0 auto; padding: 20px; background: #fff; color: #20242a; box-sizing: border-box;}
* {box-sizing: border-box;}
.jp-RenderedHTMLCommon {font-size: 17px; line-height: 1.85;}
.jp-CodeCell .cm-editor, .jp-CodeCell pre {font-size: 14px; line-height: 1.55;}
.case-navigation {font: 14px system-ui,sans-serif; padding: 10px 0 18px; border-bottom: 1px solid #ddd; margin-bottom: 20px;}
.case-navigation a {color: #15577b; margin-right: 20px;}
.case-navigation {display: flex; flex-wrap: wrap; gap: 8px 15px;}
.case-toc {font: 15px/1.7 system-ui,sans-serif; margin: 20px 0;}
.case-toc ul {padding-left: 1.4em;}
.case-toc li {margin: .2em 0;}
.reading-hint {font: 14px/1.6 system-ui,sans-serif; color: #555;}
.table-scroll {max-width: 100%; overflow-x: auto; margin: 10px 0 20px;}
.table-scroll:focus {outline: 2px solid #15577b; outline-offset: 3px;}
.jp-Cell, .jp-Cell-outputWrapper, .jp-OutputArea-child, .jp-OutputArea-output {min-width: 0; max-width: 100%;}
.jp-InputArea-editor {min-width: 0; overflow-x: auto;}
.jp-RenderedHTMLCommon pre {overflow-x: auto;}
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li {overflow-wrap: anywhere;}
@media(max-width:600px) {body {padding: 12px;} .jp-RenderedHTMLCommon {font-size: 16px;} .jp-InputPrompt, .jp-OutputPrompt {min-width: 34px; width: 34px; padding-left: 0;}}
img {max-width: 100%; height: auto;}
table {border-collapse: collapse;} th,td {padding: .4em .65em;}
</style>"""
    body = body.replace("</head>", configuration + "\n</head>")
    navigation = (
        '<nav class="case-navigation"><a href="../../index.html">课程案例首页</a>'
        '<a href="index.html">Painter 1998 案例目录</a>'
    )
    if notebook_name:
        navigation += (
            f'<a href="notebooks/{notebook_name}.ipynb" download>下载 .ipynb（含运行输出）</a>'
        )
    navigation += '<a href="report.pdf">综合报告 PDF</a>'
    if notebook_name in NOTEBOOK_NAMES:
        index = NOTEBOOK_NAMES.index(notebook_name)
        if index > 0:
            navigation += f'<a href="{NOTEBOOK_NAMES[index - 1]}.html">上一页：{NOTEBOOK_TITLES[index - 1]}</a>'
        if index < len(NOTEBOOK_NAMES) - 1:
            navigation += f'<a href="{NOTEBOOK_NAMES[index + 1]}.html">下一页：{NOTEBOOK_TITLES[index + 1]}</a>'
    navigation += "</nav>"
    for name in NOTEBOOK_NAMES:
        body = body.replace(f'href="{name}.ipynb"', f'href="{name}.html"')
    body = re.sub(r"(<body\b[^>]*>)", r"\1" + navigation, body, count=1, flags=re.IGNORECASE)
    soup = BeautifulSoup(body, "html.parser")
    soup.html["lang"] = "zh-CN"
    if soup.title is None:
        soup.head.append(soup.new_tag("title"))
    title = (
        NOTEBOOK_TITLES[NOTEBOOK_NAMES.index(notebook_name)]
        if notebook_name in NOTEBOOK_NAMES
        else "Painter 教学案例"
    )
    soup.title.string = title + " · 现代遥感技术与应用"
    icon = soup.new_tag("link", rel="icon", href="../../assets/favicon.svg", type="image/svg+xml")
    soup.head.append(icon)
    headings = soup.find_all(["h2", "h3"])
    toc = soup.new_tag("details", attrs={"class": "case-toc", "open": ""})
    summary = soup.new_tag("summary")
    summary.string = "本页目录"
    toc.append(summary)
    links = soup.new_tag("ul")
    for number, heading in enumerate(headings):
        identity = heading.get("id", f"section-{number + 1}")
        heading["id"] = identity
        item = soup.new_tag("li")
        anchor = soup.new_tag("a", href="#" + identity)
        anchor.string = heading.get_text(" ", strip=True).rstrip("¶")
        item.append(anchor)
        links.append(item)
    toc.append(links)
    soup.select_one(".case-navigation").insert_after(toc)
    hint = soup.new_tag("p", attrs={"class": "reading-hint"})
    hint.string = "窄屏阅读时，表格可在框内横向滚动；点击表格后也可使用方向键。"
    toc.insert_after(hint)
    for table in soup.find_all("table"):
        wrapper = soup.new_tag(
            "div",
            attrs={
                "class": "table-scroll",
                "role": "region",
                "tabindex": "0",
                "aria-label": "数据表，可横向滚动查看完整列",
            },
        )
        table.wrap(wrapper)
    images = soup.find_all("img")
    if image_descriptions is not None:
        if len(images) != len(image_descriptions):
            raise ValueError("Every notebook figure must have one matching description")
        for picture, description in zip(images, image_descriptions):
            picture["alt"] = description
    body = str(soup)
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


def validate_output(output: Path) -> Path:
    """限制站点输出位置，避免覆盖源码、数据和静态网页源。"""
    output = output.resolve()
    if REPOSITORY.is_relative_to(output):
        raise ValueError("Site output cannot contain the repository")
    for internal in (
        CASE / "build/notebooks",
        CASE / "build/workspace",
        CASE / "build/offline_guard",
    ):
        if output.is_relative_to(internal) or internal.is_relative_to(output):
            raise ValueError("Site output must not overlap kernels or interactive workspace")
    if output.is_relative_to(REPOSITORY) and not any(
        output.is_relative_to(root) for root in (REPOSITORY / "build", CASE / "build")
    ):
        raise ValueError("Use an ignored build directory for site output inside the repository")
    return output


def prepare_workspace() -> Path:
    """创建供交互学习的源码副本；已存在的练习文件保留。"""
    workspace = CASE / "build/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in NOTEBOOK_NAMES:
        target = workspace / (name + ".ipynb")
        if not target.exists():
            shutil.copy2(CASE / "notebooks" / target.name, target)
    return workspace


def check_previous_output(output: Path) -> None:
    """只替换有清单且未经手工修改的构建产物，保留未知文件。"""
    if not output.exists() or not any(output.iterdir()):
        return
    existing = output / "site_manifest.json"
    if not existing.is_file():
        raise ValueError("Nonempty output has no site manifest; choose an empty directory")
    manifest = json.loads(existing.read_text(encoding="utf-8"))
    known = set(manifest) | {"site_manifest.json"}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != known:
        raise ValueError("Output contains files outside the previous site manifest; preserved")
    for relative, expected in manifest.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or path.is_symlink():
            raise ValueError("Previous manifest contains an unsafe artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Previous build was edited; choose a new output to preserve it")


def build(output: Path, timeout: int = 300) -> dict:
    """在同盘暂存目录完成执行和校验，成功后替换旧站点；失败保留旧结果。

    暂存目录只由本函数创建。旧目录必须符合自己的哈希清单才允许替换，
    避免清理学习笔记。替换前删除的源页面不会遗留在新的文件树中。
    """
    from scripts.verify_site import verify

    output = validate_output(output)
    check_previous_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".site-stage-", dir=output.parent) as temporary:
        temporary_root = Path(temporary).resolve()
        staging, backup = temporary_root / "site", temporary_root / "previous-site"
        result = render_site(staging, timeout)
        verify(staging)
        had_output = output.exists()
        if had_output:
            output.replace(backup)
        try:
            staging.replace(output)
        except OSError:
            if had_output and not output.exists():
                backup.replace(output)
            raise
    return result


def render_site(output: Path, timeout: int) -> dict:
    """在新建的暂存目录生成完整站点，不向旧产物目录复制增量文件。"""
    assets = MATHJAX_CACHE
    if not all((assets / name).is_file() for name in ("tex-svg-full.js", "LICENSE", "SOURCE.json")):
        raise FileNotFoundError(
            "Missing cached MathJax assets; run build_notebooks.py --prepare-assets once"
        )
    metadata = json.loads(MATHJAX_SOURCE.read_text(encoding="utf-8"))
    if json.loads((assets / "SOURCE.json").read_text(encoding="utf-8")) != metadata:
        raise ValueError("Cached MathJax source lock differs from the repository")
    for name, expected in metadata["files"].items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"MathJax asset checksum mismatch: {name}")
    shutil.copytree(SITE_SOURCE, output)
    case_output = output / CASE_URL
    shutil.copytree(assets, case_output / "assets/mathjax", dirs_exist_ok=True)
    # 报告的唯一源在作业目录，站点只保留本次构建的副本。
    shutil.copy2(CASE / "reports/painter1998-integrated-report.pdf", case_output / "report.pdf")
    build_directory = CASE / "build/notebooks"
    build_directory.mkdir(parents=True, exist_ok=True)
    exporter = HTMLExporter(template_name="lab", embed_images=True)
    exporter.exclude_input_prompt = False
    exporter.exclude_output_prompt = False
    exporter.mathjax_url = "assets/mathjax/tex-svg-full.js"
    records = []
    for name in NOTEBOOK_NAMES:
        path = CASE / "notebooks" / (name + ".ipynb")
        notebook, elapsed = execute_notebook(path, build_directory, timeout)
        download_directory = case_output / "notebooks"
        download_directory.mkdir(exist_ok=True)
        nbformat.write(notebook, download_directory / path.name)
        body, _ = exporter.from_notebook_node(notebook)
        descriptions = []
        for cell in notebook.cells:
            images = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
            if len(images) != len(cell.metadata.get("figure_alt", [])):
                raise ValueError(
                    f"Figure descriptions missing in {path.name}, cell {cell.get('id')}"
                )
            descriptions.extend(cell.metadata.get("figure_alt", []))
            if cell.metadata.get("site_preview"):
                if len(images) != 1:
                    raise ValueError("A site preview cell must output exactly one PNG figure")
                preview = case_output / "assets/previews" / cell.metadata["site_preview"]
                if not preview.resolve().is_relative_to(case_output.resolve()):
                    raise ValueError("Unsafe preview output name")
                preview.parent.mkdir(parents=True, exist_ok=True)
                preview.write_bytes(base64.b64decode(images[0]["data"]["image/png"]))
        html = offline_html(body, name, descriptions)
        html_path = case_output / (name + ".html")
        html_path.write_text(html, encoding="utf-8", newline="\n")
        record = {
            "notebook": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "executed_sha256": hashlib.sha256(
                (download_directory / path.name).read_bytes()
            ).hexdigest(),
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
    (case_output / "build_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "site_manifest.json"
    }
    (output / "site_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "build/site",
        help="Complete generated site root (default: repository/build/site)",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Download the fixed public MathJax package and verify npm integrity",
    )
    parser.add_argument(
        "--prepare-workspace",
        action="store_true",
        help="Create non-overwriting interactive copies in build/workspace",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.prepare_assets:
        prepare_mathjax(MATHJAX_CACHE)
    elif args.prepare_workspace:
        print("Interactive notebooks: build/workspace (existing files preserved)")
        prepare_workspace()
    else:
        build(args.output, args.timeout)


if __name__ == "__main__":
    main()
