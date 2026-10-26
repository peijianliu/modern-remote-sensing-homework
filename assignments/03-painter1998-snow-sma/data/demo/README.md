# 教学数据与字段

数据来自公开 AVIRIS、MOD09GA、USGS 光谱库以及开放许可的雪覆盖参考资料。`manifest.json` 保存来源、版本、处理配置、文件字节数和 SHA-256。本目录不包含原始整景影像、JHU/ECOSTRESS 原谱或完整 SNICAR 光学属性数据库。

## 正演与光谱

- `forward_models.mat`：`matlab/run_forward.m` 计算的 SNICAR 与 WW80 缓存。半径为 30–1500 μm、步长 10 μm，波长为 0.205–4.995 μm、步长 0.01 μm；太阳天顶角为 0°、60°、22°；两种冰光学选项为 Warren 1984 和 Warren–Brandt 2008。WW80 在默认 Notebook 中由散射参数完整重算，SNICAR-AD 保留参考缓存；完整 adding-doubling 重算使用 MATLAB。
- `scattering_inputs.npz`：从官方 202105 库提取的 `(148 radius, 480 wavelength, 2 ice_options)` 散射属性，`ss_alb` float64、`asm_prm` float32，无量纲；波长 m→μm。`scattering_inputs.source.json` 保留作者、版本、官方来源、每个 NetCDF 哈希与修改说明。用于 NumPy WW80 重算，不表示重新计算了 Mie 属性。
- `quartz_hs32.5854.asc`：公开 USGS Quartz HS32 光谱。文件头注明波长与反射率单位，读取后转换为 μm 和无量纲反射率。
- `aviris_metadata.json`：2013 年 AVIRIS 的波长中心、半高全宽、投影和原始头信息。涉及原始影像的相对路径供完整流程使用。

## 新增 AVIRIS 受控实验

`aviris_case.npz` 可用 `numpy.load(..., allow_pickle=False)` 读取。

| 字段 | 形状或含义 |
|---|---|
| `wavelength_um`、`radii_um` | 182 个实景有效波段；30–1100 μm 候选半径 |
| `snow_spectra` | 候选半径 × 182 波段，22° SNICAR 雪谱 |
| `backgrounds`、`background_names` | 182 波段 × 3 个影像背景；植被优势、亮裸地、暗裸地示例 |
| `background_row_col` | 三条背景在 30 m ROI 中的行列号，零起算 |
| `representative_spectra`、`representative_row_col` | 雪与三种背景的四条代表光谱和位置 |
| `rgb`、`valid_mask` | 统一 `/0.65` 拉伸的 RGB 图和有效像元掩膜 |
| `fixed_radii_um` | 50、100、200、500、1000 μm |
| `fixed_raw_snow`、`fixed_snow_fraction` | 五种固定半径的原始雪系数、阴影归一化雪比例 |
| `fixed_rms` | 五种固定半径的光谱 RMS，反射率单位 |
| `physical_snow_fraction`、`physical_radius_um`、`physical_rms` | 物理雪库 MGS 最小 RMS 结果 |
| `image_snow_fraction`、`image_rms` | 五种候选影像雪谱 MGS 最小 RMS 结果 |
| `image_snow_spectra`、`image_endmember_row_col` | 五条候选雪谱和对应行列号 |
| `image_absorption_area_um` | 950–1090 nm 连续统去除吸收面积，μm |
| `fcls_snow_fraction` | 同一影像背景与物理雪库的 FCLS 归一化雪比例 |

原始系数用于重建光谱与残差；阴影归一化比例用于覆盖结果对比。图层中的 NaN 表示无效或未通过归一化条件的输出。影像候选雪谱按吸收面积分位选择，编号表示谱型，未进行吸收面积到半径的标定。背景是有明确位置的代表混合谱，类别名称用于教学说明。

`representative_scan.npz` 保存五个固定半径对四条代表光谱的完整解，包括 `fractions`、`normalized`、`shade`、`rms`、`condition`、`qa`、`modeled`、`residual`。`E_c = observed − modeled`，RMS 为逐波段残差的均方根。

`fixed_radius_summary.csv` 按半径汇总共同像元上的系数、比例和光谱 RMS；`image_endmembers.csv` 列出五条雪谱的分位、位置和吸收面积；`image_endmember_summary.csv` 比较相同背景下的物理雪库与影像雪谱。`spatial_metadata.json` 给出 CRS、仿射变换、形状、选样规则及正演太阳角设定。

## MODIS 原生 1 km 单元

`modis_1km_units.csv` 每行对应一个原始状态单元，唯一键为 `roi + tile_row_1km + tile_column_1km`。`flag12` 为 MOD35 snow/ice，`flag15` 为 internal snow mask。`aviris_fsca` 与 `modis_fsca` 分别为新影像背景 FCLS 模型聚合后的雪比例。`*_valid_area_fraction` 为有效模型输出覆盖目标单元的面积比例；`common_valid` 要求两者均达到 95%。复制到 500 m 的标志不增加独立统计单元。

`product_consistency.csv` 按区域、模型、参照标志和阈值（0.1、0.5、0.9）列出 TP、FN、FP、TN、参照召回率和特异度。缺少某类时对应比率为空，`n=0` 表示严格共同掩膜内没有可比单元。这些指标描述产品一致性。

## 已归档的独立验证与尺度统计

`*_corrected_coverage.csv` 与 `same_dictionary_control_corrected.csv` 为已有面积支持规则修正后的尺度结果。两份 `*_sample_pixels.csv.gz` 与 `stratified_metrics.csv` 为 Landsat/WorldView 两场独立覆盖参考对照，参考来源为 Stillinger 和 Bair 的 CC BY 4.0 数据集（DOI: 10.5281/zenodo.4031446）。这些文件属于 `legacy_library_output`：沿用原报告的历史模型结果，与本轮采用影像背景的新实验分别展示。

WorldView 表的 `reference_fsca` 为参考雪覆盖比例，`stratum` 划分无雪、混合雪和近纯雪；方法列为模型雪比例。`common_four_methods_normalization_valid` 指四种方法共同具有有效归一化比例，统计时必须应用同一掩膜。2016 配对为同日，2017 配对相差一天。

两份样本表采用 `gzip.compress(..., compresslevel=6, mtime=0)` 无损压缩，全部 12,000 行、44 列分别保留。共 7,344,001 B 压缩为 1,881,183 B；压缩文件与解压原文 SHA-256 同时保存在 manifest 中。`pandas.read_csv` 自动识别 gzip，读取接口拒绝同一场景 CSV/GZ 并存，避免重复计数。原始计算的 `code_sha256` 保留生成时来源，第二版仅新增输入提取与无损打包记录。

同一唯一 1 km 集合的筛选账本为 420 候选→420 有效状态→100 AVIRIS 有效面积≥95%→17 共同有效面积≥95%。bit 12 阳性为 32→32→0→0，bit 15 为 119→119→20→0。17 个阴性单元仍保留；85 个历史 500 m 配对独立统计，不接入该漏斗。
