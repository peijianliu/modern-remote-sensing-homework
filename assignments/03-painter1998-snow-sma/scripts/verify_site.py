"""核对 Notebook 执行输出、HTML 本地资源、页面链接和公开信息。"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import nbformat

CASE = Path(__file__).resolve().parents[1]
REPOSITORY = CASE.parents[1]
NAMES = ("01_snow_optics", "02_fig3_sma", "03_aviris_modis")


class PageLinks(HTMLParser):
    """收集页面实际加载的资源和用户可以点击的链接。"""

    def __init__(self):
        super().__init__()
        self.resources = []
        self.anchors = []

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag in {"script", "img", "iframe", "source"} and values.get("src"):
            self.resources.append(values["src"])
        if tag == "link" and values.get("href"):
            self.resources.append(values["href"])
        if tag == "a" and values.get("href"):
            self.anchors.append(values["href"])


def local_target(page: Path, value: str) -> Path | None:
    """将本地相对 URL 转为文件路径；外链和片段标识不参与存在性检查。"""
    url = urlparse(value)
    if url.scheme or url.netloc or not url.path:
        return None
    return (page.parent / unquote(url.path)).resolve()


def verify(output: Path) -> dict:
    """检查三份完整导出，不以 HTML 文件存在代替执行与资源检查。"""
    site_root = output.resolve()
    manifest = json.loads((site_root / "site_manifest.json").read_text(encoding="utf-8"))
    actual = {
        p.relative_to(site_root).as_posix()
        for p in site_root.rglob("*")
        if p.is_file() and p.name != "site_manifest.json"
    }
    if set(manifest) != actual:
        raise ValueError("Site manifest and artifact file set differ")
    for relative, checksum in manifest.items():
        target = (site_root / relative).resolve()
        if (
            not target.is_relative_to(site_root)
            or hashlib.sha256(target.read_bytes()).hexdigest() != checksum
        ):
            raise ValueError(f"Site artifact checksum mismatch: {relative}")
    output = site_root / "assignments/03-painter1998"
    if (output / "report.pdf").read_bytes() != (
        CASE / "reports/painter1998-integrated-report.pdf"
    ).read_bytes():
        raise ValueError("Generated report differs from its canonical source")
    privacy = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/|2026E\d{8,}")
    records = []
    for name in NAMES:
        source = CASE / "notebooks" / (name + ".ipynb")
        canonical = nbformat.read(source, as_version=4)
        if any(
            c.get("outputs") or c.get("execution_count") is not None
            for c in canonical.cells
            if c.cell_type == "code"
        ):
            raise ValueError(f"Notebook source contains execution artifacts: {source.name}")
        notebook = nbformat.read(output / "notebooks" / source.name, as_version=4)
        if [c.source for c in canonical.cells] != [c.source for c in notebook.cells]:
            raise ValueError(f"Executed notebook source differs: {source.name}")
        counts = []
        for cell in notebook.cells:
            if privacy.search(cell.source):
                raise ValueError(f"Personal path or student identifier in {source.name}")
            if cell.cell_type == "code":
                if cell.execution_count is None:
                    raise ValueError(f"Unexecuted cell in {source.name}")
                counts.append(cell.execution_count)
                for item in cell.outputs:
                    if item.output_type == "error":
                        raise ValueError(f"Error output in {source.name}")
                    text = item.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if privacy.search(text):
                        raise ValueError(f"Personal path in output: {source.name}")
        if counts != list(range(1, len(counts) + 1)):
            raise ValueError(f"Nonsequential execution counts in {source.name}")
        page = output / (name + ".html")
        body = page.read_text(encoding="utf-8")
        parser = PageLinks()
        parser.feed(body)
        if any(
            urlparse(value).scheme in {"http", "https"} or value.startswith("//")
            for value in parser.resources
        ):
            raise ValueError(f"Remote rendering resource in {page.name}")
        if re.search(r"url\(\s*['\"]?(?:https?:)?//", body, re.I):
            raise ValueError(f"Remote CSS resource in {page.name}")
        targets = [local_target(page, value) for value in parser.resources + parser.anchors]
        missing = [target for target in targets if target is not None and not target.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Broken local links in {page.name}: "
                + ", ".join(target.name for target in missing)
            )
        if "assets/mathjax/tex-svg-full.js" not in parser.resources:
            raise ValueError(f"Local formula renderer absent from {page.name}")
        records.append(
            {
                "page": page.name,
                "executed_code_cells": len(counts),
                "local_links_checked": sum(target is not None for target in targets),
                "embedded_images": sum(
                    value.startswith("data:image/") for value in parser.resources
                ),
            }
        )
    for page in (site_root / "index.html", output / "index.html"):
        parser = PageLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        for value in parser.resources + parser.anchors:
            target = local_target(page, value)
            if target is not None and (
                not target.is_relative_to(site_root) or not target.is_file()
            ):
                raise FileNotFoundError(f"Broken static page link: {value}")
    return {"status": "passed", "pages": records, "artifact_files": len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "build/site")
    args = parser.parse_args()
    print(json.dumps(verify(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
