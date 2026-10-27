"""公式组件按来源锁准备，完整参考不会意外覆盖已有文件。"""

import base64
import hashlib
from io import BytesIO
import json
import tarfile

import pytest

from scripts import build_notebooks as builder
from scripts.prepare_forward_reference import prepare_reference


def mathjax_fixture(tmp_path, monkeypatch):
    contents = {"tex-svg-full.js": b"fixture script", "LICENSE": b"fixture license"}
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in contents.items():
            member = tarfile.TarInfo(
                "package/es5/" + name if name.endswith(".js") else "package/LICENSE"
            )
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    payload = buffer.getvalue()
    lock = {
        "version": "3.2.2",
        "url": "https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz",
        "npm_integrity": "sha512-"
        + base64.b64encode(hashlib.sha512(payload).digest()).decode("ascii"),
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
    }
    source = tmp_path / "SOURCE.json"
    source.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(builder, "MATHJAX_SOURCE", source)
    monkeypatch.setattr(builder, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    return source, lock, contents


def test_assets_are_prepared_without_rewriting_source_lock(tmp_path, monkeypatch):
    source, lock, contents = mathjax_fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    cache = tmp_path / "cache"
    builder.prepare_mathjax(cache)
    assert source.read_bytes() == original
    for name, payload in contents.items():
        assert (cache / name).read_bytes() == payload
    assert json.loads((cache / "SOURCE.json").read_text(encoding="utf-8")) == lock


def test_modified_package_or_extracted_bytes_are_rejected(tmp_path, monkeypatch):
    source, lock, _ = mathjax_fixture(tmp_path, monkeypatch)
    lock["files"]["tex-svg-full.js"] = "0" * 64
    source.write_text(json.dumps(lock), encoding="utf-8")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.prepare_mathjax(cache)
    assert not list(cache.iterdir())
    lock["npm_integrity"] = "sha512-not-the-package"
    source.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        builder.prepare_mathjax(cache)


def test_reference_registration_preserves_existing_output(tmp_path):
    output = tmp_path / "reference.mat"
    output.write_bytes(b"my research reference")
    with pytest.raises(FileExistsError):
        prepare_reference(output, tmp_path / "unneeded.mat")
    assert output.read_bytes() == b"my research reference"


def test_reference_requires_explicit_inputs(tmp_path):
    with pytest.raises(ValueError, match="source-mat"):
        prepare_reference(tmp_path / "reference.mat")
