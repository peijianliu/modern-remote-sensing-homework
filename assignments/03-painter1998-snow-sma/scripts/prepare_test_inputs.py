"""将仓库内教学文件映射到历史测试所需路径，不访问网络。"""

from pathlib import Path
import json
import shutil


def prepare() -> None:
    root = Path(__file__).resolve().parents[1]
    pairs = {
        "forward_models.mat": "data/cache/forward_models.mat",
        "quartz_hs32.5854.asc": "data/raw/spectra/quartz_hs32.5854.asc",
        "aviris_metadata.json": "data/cache/aviris_metadata.json",
    }
    for name, destination in pairs.items():
        source = root / "data/demo" / name
        target = root / destination
        if not source.is_file():
            raise FileNotFoundError(f"Missing teaching input: {name}")
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                # 来源元数据允许 Windows 与 POSIX 相对路径分隔符不同。
                if target.suffix == ".json":

                    def normalized(value):
                        if isinstance(value, dict):
                            return {key: normalized(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [normalized(item) for item in value]
                        return value.replace("\\", "/") if isinstance(value, str) else value

                    old = normalized(json.loads(target.read_text(encoding="utf-8")))
                    new = normalized(json.loads(source.read_text(encoding="utf-8")))
                    if old == new:
                        continue
                raise FileExistsError(f"Different cache exists; preserve it: {destination}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


if __name__ == "__main__":
    prepare()
