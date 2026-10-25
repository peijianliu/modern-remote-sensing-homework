# Painter 1998 雪粒径与光谱混合分析

这个案例研究雪粒半径如何改变反射率，以及这种变化如何影响混合像元的积雪比例估计。提供三份可执行 Notebook、完整运行输出、HTML 和综合实验报告。

## 先阅读，再运行

- [在线 HTML](https://peijianliu.github.io/modern-remote-sensing-homework/assignments/03-painter1998/)
- [综合报告 PDF](reports/painter1998-integrated-report.pdf)
- [Notebook 源文件](notebooks/)
- [数据字段与来源](data/demo/README.md)
- [代码与图表索引](CODE_MAP.md)
- [公开数据与使用条件](PUBLIC_DATA_POLICY.md)

Notebook 01 解释冻结正演结果；02 重新合成和解混 14,746 组光谱；03 重算真实像元、产品一致性和分层统计。教学数据随仓库提供，默认流程无需 MATLAB、Earthdata 登录或完整遥感影像。

## 快速开始

使用 Python 3.11，在本作业目录运行以下 PowerShell 命令：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[notebook,dev]"
& .\.venv\Scripts\python.exe scripts/build_notebooks.py
```

macOS / Linux 将解释器替换为 `.venv/bin/python`。构建脚本在干净内核中执行全部 Notebook，再生成 HTML；输出到仓库 `docs/assignments/03-painter1998/`，构建记录位于忽略提交的 `build/`。

交互阅读可运行：

```powershell
& .\.venv\Scripts\python.exe -m jupyterlab notebooks
```

无需在 Notebook 中安装依赖。MathJax 公式资源已随 HTML 保留。完整克隆后，HTML 构建与教学计算均使用本地文件。

## 三份 Notebook

| Notebook | 主要计算 | 数值检查 |
|---|---|---|
| `01_snow_optics.ipynb` | 0°/60°雪谱、粒径—反照率、两套冰光学选项、SNICAR/WW80 对照 | 波长、半径、角度与输出范围 |
| `02_fig3_sma.ipynb` | 146 半径×101 比例×224 波段；固定 200 μm、多粒径 MGS、FCLS 示例和 E_c | 已知比例恢复、匹配 RMS、残差定义 |
| `03_aviris_modis.ipynb` | 五个固定半径、五种影像雪谱、唯一 1 km 标志交叉表、WorldView 分层与尺度结果 | 共同掩膜、样本数、统计与冻结结果一致 |

雪库给出的是光学等效半径，单位 μm。MGS 保留无约束系数，阴影为 `1−sum(F)`；FCLS 约束非阴影系数非负且总和不超过 1。原始系数用于光谱重建与 `E_c=observed−modeled`；阴影归一化比例用于覆盖对比。

## 教学数据与完整处理

`data/demo/` 约 19 MB，包括完整小型正演缓存、公开 AVIRIS 代表谱与图层、石英谱、配对表及公开覆盖参考的派生结果。每个数据文件的来源和 SHA-256 见 `data/demo/manifest.json`。

本轮新增实景实验采用公开 AVIRIS 中预先选定的植被优势、亮裸地和暗裸地背景谱。历史库谱实验与新增影像背景实验分别保存和解释。JHU 原谱、EcoSIS 砾石候选、原始商业 WorldView 影像和组员原始作业不进入教学数据。

完整处理保留在 `snowmix/image.py`、`scripts/acquire_*.py`、`matlab/` 和配置中。需要完整数据时，先阅读 [输入来源](DATA_SOURCES.md)，安装 `.[full]`，从官方来源准备原始影像和 MATLAB 正演缓存；不要将整景数据、登录信息或运行日志加入 Git。

## 测试

默认核心与教学测试：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m "not requires_gis and not full_data"
```

包括原有 GIS 与缓存参数回归的完整单元测试：

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[full]"
& .\.venv\Scripts\python.exe scripts/prepare_test_inputs.py
& .\.venv\Scripts\python.exe -m pytest -q
```

`prepare_test_inputs.py` 只把仓库内教学文件复制到历史测试路径，不下载数据、不覆盖不同内容的缓存。病态端元测试同时检查系数误差和反射率 RMS，容纳不同 BLAS 实现的浮点累加差异。CI 在 Windows/Linux 执行 Notebook 与 HTML 构建，并运行数值/GIS 测试。

`pyproject.toml` 固定主要依赖；`requirements-lock.txt` 记录本次完整测试环境和平台条件。

## 引用与贡献

核心文献：Painter et al. (1998), *The effect of grain size on spectral mixture analysis of snow-covered area from AVIRIS data*, Remote Sensing of Environment, DOI: [10.1016/S0034-4257(98)00041-8](https://doi.org/10.1016/S0034-4257(98)00041-8)。

案例以刘培剑的计算为主线，结合钟超凡、高韵淇、徐郅杰、黄磊的作业设计，详见仓库根目录 `CONTRIBUTORS.md`。源码 MIT；原创教学文字和报告 CC BY 4.0；外部模型与数据分别遵循 [第三方许可](THIRD_PARTY.md)。
