# Painter 1998 雪粒径与光谱混合分析

同样的雪面积可以因为粒径变化呈现不同光谱。这个案例从一个波长的手算出发，检查固定雪端元怎样影响覆盖估计，再把解混、空间聚合和验证设计连接起来。面向已有 Python、线性代数基础的同学。

## 两条学习路线

- **30 分钟阅读**：打开 [HTML 01](https://peijianliu.github.io/modern-remote-sensing-homework/assignments/03-painter1998/01_snow_optics.html)，看手算与粒径光谱图；再读 [综合报告](reports/painter1998-integrated-report.pdf) 的 Fig. 3 控制实验，比较雪系数与光谱 RMS。
- **3 小时实践**：克隆仓库，在干净 Python 3.11 环境构建三份 Notebook；随后创建工作副本，逐页预测、手算并改参数。保留一份你自己的“输入改了什么、输出为何变化”的记录。

[在线目录](https://peijianliu.github.io/modern-remote-sensing-homework/assignments/03-painter1998/) · [源码 Notebook](notebooks/) · [数据字段](data/demo/README.md) · [图表和代码索引](CODE_MAP.md) · [数据使用条件](PUBLIC_DATA_POLICY.md)

## 快速开始

在仓库本作业目录运行以下 PowerShell 命令。环境隔离不继承 Anaconda 的包：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[notebook,dev]"
& .\.venv\Scripts\python.exe scripts/build_notebooks.py --prepare-assets
& .\.venv\Scripts\python.exe scripts/build_notebooks.py
& .\.venv\Scripts\python.exe scripts/verify_site.py
```

macOS / Linux 使用 `.venv/bin/python` 替换解释器路径。安装依赖需要联网；安装完成后首次执行 `--prepare-assets`，从官方 npm 下载固定 MathJax 并按版本化 SHA-512/SHA-256 核对，写入忽略的 `build/assets/`。之后默认教学计算和 HTML 导出使用本地数据、模板及缓存组件，无需 MATLAB、Earthdata 登录或大型影像。

默认完整站点在仓库根 `build/site/`，从 `index.html` 打开。HTML 中的下载链接提供**已执行 Notebook**；Git 中的 `notebooks/` 仅保留无输出源码。运行时间与哈希保存在生成站点的 `build_manifest.json`，源文件不会因构建改变。

交互学习先创建可修改副本，再打开 JupyterLab：

```powershell
& .\.venv\Scripts\python.exe scripts/build_notebooks.py --prepare-workspace
& .\.venv\Scripts\python.exe -m jupyterlab build/workspace
```

工作副本位于忽略提交的 `build/workspace/`，重复准备会保留已有练习。下载的单本 Notebook 仍需完整仓库中的共享模块和教学数据。

`--output` 指定**完整站点根目录**，例如 `--output ../../build/my-site`；不能指向源码、数据或静态网页源目录。`--prepare-assets` 是首次准备与缓存缺失时的显式联网步骤；普通构建不联网，也不回写 `site/`。离线阅读包已包含完整组件。

## 运行后应该看到什么

| Notebook | 预期首行与关键结果 | 图件 | Windows 实测执行时间¹ |
|---|---|---|---|
| 01 雪光学 | `光谱数组：148 半径 × 480 波长 × 3 角度 × 2 光学选项`；现算 426,240 值；11,520 独立参考值及加权校核最大差 <1e-6 | 单谱、0°/60°、粒径—加权反照率、模型对照 | 2.76 s |
| 02 混合分析 | `三波段示例：观测 [0.62, 0.44, 0.20]`；14,746 条光谱，匹配 RMS <1e-8 | Fig. 3、重建和残差、FCLS 对照 | 7.47 s |
| 03 验证设计 | `教学数据：62,908 个有效 30 m 像元；420 个候选唯一 1 km 单元`；筛选 420→100→17 | 地图、分母诊断、配对区间、面积门槛与分层统计 | 3.67 s |

¹ Python 3.11.7，单次本地实测，包括内核启动；HTML 导出另有开销。以自己构建清单中的耗时为准。固定基线断言和改参数练习分开，方便比较输入输出。

**字体 FAQ**：检测顺序为 Noto Sans CJK SC、Source Han Sans SC、Microsoft YaHei、SimHei。缺少这些字体时，图内标签回退英文并打印提示，中文教学正文保留。公式 SVG 使用本地 MathJax，不依赖微软字体。GitHub Markdown 的字号由 GitHub 控制；导出 HTML 的正文为 17 px。

## 数据与数学约定

`data/demo/` 包括约 1.36 MB 官方散射参数子集、独立太阳谱与约 183 KB 参考子集、公开 AVIRIS 派生图层、石英谱、配对表和无损压缩的独立覆盖参考表。每项来源、精度转换及 SHA-256 见 [数据说明](data/demo/README.md) 和 `manifest.json`。完整商业 WorldView 影像、未确认条件的 ECOSTRESS/EcoSIS 原谱与五份原始作业不随仓库分发。

WW80 使用半无限雪层、直接入射的 delta-Eddington 闭式解。NumPy 从散射输入计算全部 426,240 个 `ww_albedo` 值和太阳谱加权结果，物理图和 Fig. 3 均使用这些现算结果。仅独立校核 cell 读取四种半径的 MATLAB 参考子集和 SNICAR-AD 加权结果；完整 MAT 不随仓库分发。历史全网格校核最大差为 5.16e-8，当前参考子集的校核在默认流程实际运行。散射属性从官方库提取，未在本项目重新计算 Mie 属性。

MGS 使用无约束原始系数，阴影为 `1-sum(F)`；FCLS 对非阴影系数施加非负、总和不超过 1 的约束。原始系数重建光谱，`E_c=observed-modeled`；阴影归一化比例用于覆盖对照。真实模型的比较先固定共同有效域。

## 完整处理与参数包重建

[输入来源](DATA_SOURCES.md) 说明完整 AVIRIS/MODIS 流程。安装 `.[full]` 后，独立入口可处理原始影像、投影和整景解混；`prepare_case_data.py --source-root` 还需已归档的尺度/WorldView 统计，默认教学不调用它。

如果要从上游输入重建小型散射参数包，从 [官方光学库说明](https://github.com/mflanner/SNICARv3/blob/master/Links%20to%20optics%20libraries.md) 获取 202105 库，按 `data/sources/optics.json` 核对约 500 MB 归档的 SHA-256。解压后运行：

```powershell
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[full]"
& .\.venv\Scripts\python.exe scripts/prepare_teaching_optics.py --prepare-optics --optics-root data/cache/optics/snicar_480band --output build/scattering_inputs.npz
```

输出以 `ss_alb` float64、`asm_prm` float32 存储，附逐个 NetCDF 文件哈希和单位说明；已有不同内容不会覆盖。MATLAB 参考入口用显式 34 字段构造器 `pure_snow_inputs`，`verify_forward` 对照四种半径、两个角度、两个冰选项共 16 个结果。第三方源码保持冻结。

完整 MATLAB 参考可在已准备官方光学库的工作区显式生成：

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_forward_reference.py --prepare-reference --workspace <prepared-workspace> --matlab <matlab-executable>
```

也可用 `--source-mat <existing-forward-models.mat>` 登记已有完整结果。两种方式均写入忽略的 `data/cache/`，已有文件保留。它们用于可选全网格校核及历史完整处理；默认三本 Notebook 使用公开小型输入。

固定半径对照的 `fixed_radius_uncertainty.csv` 给出分母分布、全输入达标率和均值95%区间；区间由30个1.5 km区块、1000次配对重采样得到。`coverage_sensitivity.csv` 比较80%、90%、95%有效面积门槛，三档最终均为17个共同阴性1 km单元。

## 检查

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m "not requires_gis and not full_data"
& .\.venv\Scripts\python.exe scripts/check_reproducibility.py
```

后者连续构建两次，逐次核对整个仓库的跟踪文件哈希与 Git 状态；第二次禁止 Python 外网连接，保留本地内核通信。执行结果保存在忽略的 `build/`，不会回写源码。CI 在 Windows/Linux 运行相同检查；Pages 部署已通过检查的 Ubuntu artifact。

完整 GIS 回归需安装 `.[full]`，运行 `scripts/prepare_test_inputs.py` 后执行 `python -m pytest -q`。标记为 `full_data` 的历史缓存配置测试只在完整 MATLAB 参考存在时运行；干净教学检出会明确跳过该项。主要依赖由 `pyproject.toml` 固定，完整环境约束见 `requirements-lock.txt`。

## 引用与贡献

核心论文：Painter et al. (1998), *The effect of grain size on spectral mixture analysis of snow-covered area from AVIRIS data*, Remote Sensing of Environment, DOI [10.1016/S0034-4257(98)00041-8](https://doi.org/10.1016/S0034-4257(98)00041-8)。案例引用信息见根目录 `CITATION.cff`。

刘培剑整理与维护，结合钟超凡、高韵淇、徐郅杰、黄磊的方法设计，详见 `CONTRIBUTORS.md`。原创代码 MIT，原创报告和教学说明 CC BY 4.0；外部代码、数据保留各自使用条件和署名，见 [第三方来源](THIRD_PARTY.md)。

教学组织参考 [NEON 单像元光谱](https://www.neonscience.org/resources/learning-hub/tutorials/plot-refl-spectra-py)、[Software Carpentry NumPy](https://swcarpentry.github.io/python-novice-inflammation/02-numpy.html)、[Earth Lab 掩膜课程](https://earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/landsat-in-Python/remove-clouds-from-landsat-data/) 和 [Project Pythia 重网格工作流](https://projectpythia.org/cmip6-cookbook/notebooks/example-workflows/xesmf-ohu/)；采用问题导入、手算预期、筛选账本和科学步骤衔接的组织方式。
