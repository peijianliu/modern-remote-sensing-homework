"""提取正演所需的官方光学属性，保留第三方源码原件。"""

from pathlib import Path
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "data/cache/optics"
destination.mkdir(parents=True, exist_ok=True)
count = 0
with tarfile.open(ROOT / "data/raw/snicar_v3_optics.tar.gz", "r|gz") as archive:
    for member in archive:
        if not member.isfile():
            continue
        # 官方 SNICAR 仍会读取浓度为零的组分属性。
        if "algae" in member.name.lower():
            continue
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"Unsafe archive member: {member.name}")
        if target.exists():
            if target.stat().st_size != member.size:
                raise ValueError(f"Existing extracted file differs: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = archive.extractfile(member)
        with target.open("xb") as handle:
            while chunk := stream.read(2**20):
                handle.write(chunk)
        count += 1

optics_roots = list(destination.rglob("ice_Wrn84_0200.nc"))
if len(optics_roots) != 1:
    raise ValueError(f"Could not identify optics root: {optics_roots}")
optics_root = optics_roots[0].parent.parent
source = ROOT / "vendor/SNICARv3/snicarAD_v3.m"
if (
    hashlib.sha256(source.read_bytes()).hexdigest()
    != "f1affb2ef5a8755ebaef7f91296607ce9a887bc66df0e0880d55d9667d723adb"
):
    raise ValueError("Frozen SNICAR source checksum mismatch")
original = source.read_text(encoding="utf-8")
if len(re.findall(r"(?m)^dir_op_root\s*=\s*[^\n]+", original)) != 1:
    raise ValueError("Expected exactly one optical directory assignment")
relative_root = optics_root.relative_to(ROOT).as_posix() + "/"
patched, replacements = re.subn(
    r"(?m)^dir_op_root\s*=\s*[^\n]+", f"dir_op_root = '{relative_root}';", original, count=1
)
if replacements != 1:
    raise ValueError("SNICAR path patch did not match frozen source")
work = ROOT / "data/cache/matlab"
work.mkdir(parents=True, exist_ok=True)
patched_file = work / "snicarAD_v3.m"
if patched_file.exists() and patched_file.read_text(encoding="utf-8") != patched:
    raise ValueError("Different patched implementation already exists; preserved")
patched_file.write_text(patched, encoding="utf-8")
manifest = {
    "extracted_new_files": count,
    "optics_root": relative_root,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "working_copy_sha256": hashlib.sha256(patched_file.read_bytes()).hexdigest(),
    "only_patch": "dir_op_root; original frozen source preserved",
}
(ROOT / "data/sources/optics_extraction.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2), flush=True)
