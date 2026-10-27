"""用真实 Chromium 检查桌面与窄屏阅读、公式、下载和前端错误。

先安装可选 browser 依赖，并运行 python -m playwright install --no-shell chromium。
脚本只服务指定的构建目录，绑定本机回环地址；禁止页面请求外部站点。
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

PAGES = (
    "index.html",
    "assignments/03-painter1998/index.html",
    "assignments/03-painter1998/01_snow_optics.html",
    "assignments/03-painter1998/02_fig3_sma.html",
    "assignments/03-painter1998/03_aviris_modis.html",
)


class QuietHandler(SimpleHTTPRequestHandler):
    """不将本地HTTP访问写入控制台。"""

    def log_message(self, format, *args) -> None:
        return


def check_site(site: Path, report: Path, channel: str = "chromium") -> dict:
    """在375、768、1440 px宽度核验五页，并保留截图和结构化结果。"""
    if report.exists():
        raise FileExistsError("Use a new browser-check report path")
    if not (site / "site_manifest.json").is_file():
        raise FileNotFoundError("Build and verify the complete site first")
    report.parent.mkdir(parents=True, exist_ok=True)
    screenshots = report.parent / (report.stem + "_screenshots")
    screenshots.mkdir(exist_ok=False)
    handler = partial(QuietHandler, directory=str(site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records, failures = [], []
    base = f"http://127.0.0.1:{server.server_port}/"
    try:
        with sync_playwright() as playwright:
            # channel=chromium使用完整Chromium的新无头模式，不依赖headless-shell。
            browser = playwright.chromium.launch(headless=True, channel=channel)
            browser_version = browser.version
            try:
                for width in (375, 768, 1440):
                    context = browser.new_context(
                        viewport={"width": width, "height": 900}, accept_downloads=True
                    )
                    context.route(
                        "**/*",
                        lambda route: route.continue_()
                        if route.request.url.startswith(base)
                        or route.request.url.startswith("data:")
                        else route.abort(),
                    )
                    try:
                        for path in PAGES:
                            page = context.new_page()
                            errors = []
                            page.on("pageerror", lambda error: errors.append(str(error)))
                            page.on(
                                "console",
                                lambda message: errors.append(message.text)
                                if message.type == "error"
                                else None,
                            )
                            response = page.goto(base + path, wait_until="networkidle")
                            if "01_snow" in path or "02_fig" in path or "03_aviris" in path:
                                page.wait_for_function(
                                    "window.MathJax && window.MathJax.startup && window.MathJax.startup.promise"
                                )
                                page.evaluate(
                                    "async () => { await window.MathJax.startup.promise; }"
                                )
                            layout = page.evaluate("""() => ({
                                lang:document.documentElement.lang,
                                title:document.title,
                                viewport:innerWidth,
                                documentWidth:document.documentElement.scrollWidth,
                                brokenImages:[...document.images].filter(x=>!x.complete || !x.naturalWidth).length,
                                missingAlt:[...document.images].filter(x=>!x.alt || x.alt.includes('No description')).length,
                                formulaErrors:document.querySelectorAll('mjx-merror, [data-mjx-error]').length,
                                formulas:document.querySelectorAll('mjx-container[jax=\"SVG\"]').length,
                                tables:document.querySelectorAll('table').length,
                                wrappedTables:document.querySelectorAll('.table-scroll table').length
                            })""")
                            entry = {
                                "page": path,
                                "width": width,
                                "http_status": response.status if response else None,
                                **layout,
                                "javascript_errors": errors,
                            }
                            issues = []
                            if entry["http_status"] != 200:
                                issues.append("HTTP response")
                            if layout["lang"] != "zh-CN":
                                issues.append("Chinese language metadata")
                            if layout["documentWidth"] > width + 1:
                                issues.append("Page horizontal overflow")
                            if layout["brokenImages"] or layout["missingAlt"]:
                                issues.append("Figure loading or alternative text")
                            if layout["formulaErrors"] or errors:
                                issues.append("JavaScript or formula error")
                            if (
                                path.endswith(("01_snow_optics.html", "02_fig3_sma.html"))
                                and not layout["formulas"]
                            ):
                                issues.append("Missing rendered SVG formula")
                            if path.endswith(
                                ("01_snow_optics.html", "02_fig3_sma.html", "03_aviris_modis.html")
                            ):
                                if layout["tables"] != layout["wrappedTables"]:
                                    issues.append("Unwrapped table")
                                if "Notebook" == layout["title"]:
                                    issues.append("Generic page title")
                                if width == 1440:
                                    link = page.locator("nav.case-navigation a[download]")
                                    with page.expect_download() as item:
                                        link.click()
                                    download = item.value
                                    if not download.suggested_filename.endswith(".ipynb"):
                                        issues.append("Notebook filename")
                                    body = json.loads(
                                        Path(download.path()).read_text(encoding="utf-8")
                                    )
                                    if not any(c.get("outputs") for c in body["cells"]):
                                        issues.append("Notebook outputs absent")
                            pdf = page.locator('a[href$="report.pdf"]').first
                            if pdf.count():
                                href = pdf.get_attribute("href")
                                from urllib.parse import urljoin

                                result = context.request.get(urljoin(base + path, href))
                                if result.status != 200 or not result.body().startswith(b"%PDF-"):
                                    issues.append("Report link or PDF header")
                            page.screenshot(
                                path=str(
                                    screenshots
                                    / f"{width}_{path.replace('/', '_').removesuffix('.html')}.png"
                                ),
                                full_page=False,
                            )
                            entry["issues"] = issues
                            records.append(entry)
                            failures.extend(f"{path}@{width}: {issue}" for issue in issues)
                            page.close()
                    finally:
                        context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    result = {
        "status": "failed" if failures else "passed",
        "browser_channel": channel,
        "browser_version": browser_version,
        "pages": records,
        "failures": failures,
    }
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise ValueError("Browser checks failed: " + "; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--channel",
        choices=("chromium", "chrome", "msedge"),
        default="chromium",
        help="Use an installed Chrome or Edge if the test Chromium is unavailable",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check_site(args.site.resolve(), args.report.resolve(), args.channel),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
