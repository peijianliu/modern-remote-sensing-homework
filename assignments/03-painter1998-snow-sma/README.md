# Painter 1998 方法复现与现代遥感数据扩展

雪面积相同，粒径和背景不同，观测光谱也会不同。本案例从一个波长的手算开始，逐步解释光谱正演、线性解混、物理约束及独立覆盖验证，面向已有 Python、线性代数基础的同学。

- **30 分钟阅读**：[01 雪光学](https://peijianliu.github.io/modern-remote-sensing-homework/assignments/03-painter1998/01_snow_optics.html)的手算与雪谱，再看[综合报告](reports/painter1998-integrated-report.pdf)的 Fig. 3。
- **3 小时实践**：建立 Python 3.11 环境，运行三份 Notebook，在工作副本中改半径、角度和覆盖门槛，记录预测、输入输出与结果解释。

[在线目录](https://peijianliu.github.io/modern-remote-sensing-homework/assignments/03-painter1998/) · [实验分支与主要结果](EXPERIMENTS.md) · [数据字段](data/demo/README.md) · [代码索引](CODE_MAP.md) · [数据使用条件](PUBLIC_DATA_POLICY.md)

## 快速开始

从获取仓库开始，PowerShell 命令如下：

```powershell
git clone https://github.com/peijianliu/modern-remote-sensing-homework.git
cd modern-remote-sensing-homework/assignments/03-painter1998-snow-sma
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[notebook,dev]"
& .\.venv\Scripts\python.exe scripts/build_notebooks.py --prepare-assets
& .\.venv\Scripts\python.exe scripts/build_notebooks.py
& .\.venv\Scripts\python.exe scripts/verify_site.py
```

Linux / macOS 将解释器换为 `.venv/bin/python`。首次安装及 `--prepare-assets` 需要联网；MathJax 从官方 npm 下载，按固定 SHA-512/SHA-256 核对后存入忽略目录。之后默认计算和 HTML 导出使用本地数据与组件，无需 MATLAB、Earthdata 登录或大型影像。

完整站点写到仓库根 `build/site/`，从 `index.html` 打开。Git 仅保存无输出 Notebook；已执行版本、HTML、耗时和哈希都写入构建目录。构建先在新暂存目录完成执行和校验，再替换已核验的旧站点；失败会保留旧结果。手工修改或加入笔记的旧输出目录不会覆盖。

交互学习使用独立副本：

```powershell
& .\.venv\Scripts\python.exe scripts/build_notebooks.py --prepare-workspace
& .\.venv\Scripts\python.exe -m jupyterlab build/workspace
```

已有练习不会覆盖。下载的单本 Notebook 仍需完整仓库中的共享模块与教学数据。`--output` 指定完整站点根，不能指向数据、Notebook 源码或静态源 `site/`。

## 运行后看到什么

| Notebook | 首行与关键检查 | 主要图件 | Windows 实测¹ |
|---|---|---|---:|
| 01 雪光学 | `散射输入：148 半径 × 480 波长 × 2 光学选项`；角度轴在正演时加入，得到 426,240 个值 | 手算、单谱、0°/60°及加权反照率 | 2.93 s |
| 02 Fig. 3 与解混 | `三波段示例：观测 [0.62, 0.44, 0.20]`；重新生成 14,746 条混合光谱，匹配 RMS <10⁻⁸ | 固定/多粒径、残差、七种失配 | 11.97 s |
| 03 实景与尺度 | `教学数据：62,908 个有效 30 m 像元；420 条 ROI 记录；406 个原生 1 km 单元` | 系数可行性、留出谱、QA及独立覆盖 | 13.56 s |

¹ Python 3.11.7，一次实际执行，含内核启动；HTML 导出另计。自己的耗时见生成站点 `build_manifest.json`。形状、计数和指标由断言检查，打印是阅读提示。

**字体 FAQ**：优先检测 Noto Sans CJK SC、Source Han Sans SC、Microsoft YaHei、SimHei，兼容包含中文字形的 Noto Sans CJK JP。缺少支持字体时图内标签回退英文并打印提示；中文正文保留。CI 安装 Noto CJK。HTML 正文为 17 px，窄屏表格在容器内横向滚动。GitHub Markdown 字号由 GitHub 控制。

## 实验范围与关键发现

MGS 为修正 Gram–Schmidt 正交化的无约束最小二乘，保留有符号系数；FCLS 约束各端元非负、总和不超过 1，剩余部分为阴影。原始系数重建光谱，`E_c=observed-modeled`；去阴影雪比例为雪系数除以地物系数之和，分母≤0.05时留空。

2013 年当前影像背景模型的 62,861 个共同像元中，物理 MGS 约 55.43% 的雪系数为负，全部系数及阴影同时可行的比例约 1.17%。地图显式区分负值、其他越界和 NoData，并与 FCLS 比例并列。Notebook 用背景正交补投影解释负号。

匹配合成库的近零误差用于数值恢复检查。七种失配对照与独立覆盖结果显示，多粒径降低光谱 RMS 的收益随背景、场景和约束变化。2016 年新增同日 MODIS 严格七波段共同域有 1,416 个 500 m 单元，含 463 个参考雪比例≥0.1；1 km 状态对照有 408 个单元，包含 123 个参照阳性。2017 年阳性较少，分开解释。

历史库谱、当前影像背景、公开端元 Landsat 与同日 MODIS 分支分别命名，不能将其中一条的独立误差用于另一条。完整矩阵见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 三级复现

| 级别 | 输入与输出 | 核验范围 |
|---|---|---|
| L1 教学重算 | 随仓库样本 → 正演、解混、指标、图表、HTML | Windows/Linux CI；数值及浏览器检查 |
| L2 完整公开输入 | 完整 AVIRIS 重新选端元和解混；原始 MOD09GA 字段与公开参考重新聚合评价 | 两项实际端到端测试通过；HDF 产品标识、比例因子与哈希核对 |
| L3 历史全部实验 | 历史原始输入、原文条件、完整模型库及归档结果 | 部分输入未恢复；历史统计与现行计算分别标注 |

L2 从已下载输入计时：AVIRIS 重建一次约 79.33 s；两景 MODIS 独立验证准备一次约 57.77 s。没有计入网络下载、解压、HDF 字段导出或完整光学库重建。`prepare_case_data.py` 是历史材料整理入口，还需要归档统计；当前完整重建使用下面的两个入口。

完整输入按 [DATA_SOURCES.md](DATA_SOURCES.md) 准备，安装 `.[full]`：

```powershell
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[full]"
& .\.venv\Scripts\python.exe scripts/rebuild_public_case.py --source-root <public-input-root> --output build/full-aviris
& .\.venv\Scripts\python.exe scripts/prepare_modis_hdf.py --input <2016-HDF> --output <field-root>/modis_2016/modis_fields.mat --matlab <matlab-executable>
& .\.venv\Scripts\python.exe scripts/prepare_modis_hdf.py --input <2017-HDF> --output <field-root>/modis_2017/modis_fields.mat --matlab <matlab-executable>
& .\.venv\Scripts\python.exe scripts/prepare_modis_validation.py --source-root <public-input-root> --field-root <field-root> --output build/full-modis
```

HDF4 字段提取使用现有 MATLAB 入口；默认三份 Notebook 只读小表、重新解混。原生状态标志以 1 km 单元计数，反射率覆盖误差以原生 500 m 单元评价。

## 光学输入与 MATLAB 参考

WW80 从约 1.36 MB 官方散射属性和独立太阳谱重算；`ss_alb` 保留 float64，`asm_prm` 存为 float32，计算提升至 float64。散射属性由官方 202105 库提取，未重新计算 Mie 属性。默认仅独立校核 cell 读取约 183 KB MATLAB 参考子集；完整 `forward_models.mat` 不随仓库分发。

从官方约 500 MB 库重建小参数包，以及显式准备完整 MATLAB 参考的命令见 [DATA_SOURCES.md](DATA_SOURCES.md)。MATLAB 使用显式 34 字段构造器，第三方源码与许可保持冻结。

## 检查与浏览器验证

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m "not requires_gis and not full_data"
& .\.venv\Scripts\python.exe scripts/check_reproducibility.py
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[browser]"
& .\.venv\Scripts\python.exe -m playwright install --no-shell chromium
& .\.venv\Scripts\python.exe scripts/browser_smoke.py --site ../../build/site --report ../../build/browser/check.json
```

已有 Chrome/Edge 可用 `--channel chrome` 或 `--channel msedge`。浏览器检查覆盖首页、目录和三份 HTML，在 375、768、1440 px 下核对 JS、公式、图片、语言、溢出、Notebook 下载及 PDF 链接。

连续构建的第二次禁止 Python 外网，逐次比较跟踪文件哈希与 Git 状态。CI 发布已通过检查的 Ubuntu artifact。完整 GIS 测试安装 `.[full]`；`full_data` 测试需要显式设置 `PAINTER_SOURCE_ROOT` 和 `PAINTER_MODIS_FIELD_ROOT`，缺少完整输入时明确跳过。

验证完整实验批次时必须指定目录和关键阶段，空批次、缺阶段或损坏产物会失败：

```powershell
& .\.venv\Scripts\python.exe scripts/verify_artifacts.py --experiments-dir build --required-stage public_aviris_rebuild --required-stage independent_modis
```

## 来源 许可与贡献

原文 DOI：[10.1016/S0034-4257(98)00041-8](https://doi.org/10.1016/S0034-4257(98)00041-8)。USGS 松针与石英来源：[splib05a](https://doi.org/10.3133/ofr03395)。WorldView 仅采用 Stillinger 与 Bair 在 [Zenodo 4031446](https://doi.org/10.5281/zenodo.4031446) 公开发布、CC BY 4.0 授权的派生覆盖，已作裁剪、聚合、分层。

商业原始 WorldView、未确认使用条件的原谱和五份原始作业不进入公开包。下载及读取入口排除了两条未确认的历史 ECOSTRESS 原谱；历史结果只保留统计与引用。

刘培剑整理维护，结合钟超凡、高韵淇、徐郅杰、黄磊的设计，见[贡献说明](../../CONTRIBUTORS.md)。原创代码 MIT，报告和教学文档 CC BY 4.0；外部数据保留原许可与修改说明，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

教学组织参考 [NEON](https://www.neonscience.org/resources/learning-hub/tutorials/plot-refl-spectra-py)、[Software Carpentry](https://swcarpentry.github.io/python-novice-inflammation/02-numpy.html)、[Earth Lab](https://earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/remove-clouds-from-landsat-data/)和 [Project Pythia](https://projectpythia.org/cmip6-cookbook/notebooks/example-workflows/xesmf-ohu/)的问题导入、小矩阵、掩膜观察与科学步骤衔接。
