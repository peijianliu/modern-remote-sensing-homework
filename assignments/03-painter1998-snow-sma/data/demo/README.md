# 教学数据与字段

数据来自公开 AVIRIS、MOD09GA、USGS 光谱库以及开放许可的雪覆盖参考资料。`manifest.json` 保存来源、版本、处理配置、文件字节数和 SHA-256。本目录不包含原始整景影像、JHU/ECOSTRESS 原谱或完整 SNICAR 光学属性数据库。

## 正演与光谱

- `solar_illumination.npz`：SNICAR中纬度冬季直接入射太阳谱，形状为 `(3 angles, 480 wavelengths)`，从完整模型输入通量无损提取。与散射参数共同构成默认WW80正演输入，反照率现算。
- `forward_reference.npz`：四个代表半径（50、200、500、1000 μm）的独立MATLAB WW80/SNICAR-AD光谱，形状为 `(4,480,3,2)`，以及148个半径的加权参考值。仅用于校核cell。`forward_reference.source.json`说明来源、完整历史MAT哈希及子集处理。完整MAT已转为可选重建输入，不随仓库分发。
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
| `representative_spectra`、`representative_row_col` | 旧版四条自拟合控制谱及位置；后三条就是背景端元，当前教学使用另选的留出像元 |
| `rgb`、`valid_mask` | 统一 `/0.65` 拉伸的 RGB 图和有效像元掩膜 |
| `fixed_radii_um` | 50、100、200、500、1000 μm |
| `fixed_raw_snow`、`fixed_snow_fraction` | 五种固定半径的原始雪系数、阴影归一化雪比例 |
| `fixed_denominator` | 五模型的 `1−shade`，从原始非阴影系数求和直接计算，float64 |
| `fixed_rms` | 五种固定半径的光谱 RMS，反射率单位 |
| `physical_snow_fraction`、`physical_radius_um`、`physical_rms` | 物理雪库 MGS 最小 RMS 结果 |
| `image_snow_fraction`、`image_rms` | 五种候选影像雪谱 MGS 最小 RMS 结果 |
| `image_snow_spectra`、`image_endmember_row_col` | 五条候选雪谱和对应行列号 |
| `image_absorption_area_um` | 950–1090 nm 连续统去除吸收面积，μm |
| `fcls_snow_fraction` | 同一影像背景与物理雪库的 FCLS 归一化雪比例 |

原始系数用于重建光谱与残差；阴影归一化比例用于覆盖结果对比。图层中的 NaN 表示无效或未通过归一化条件的输出。影像候选雪谱按吸收面积分位选择，编号表示谱型，未进行吸收面积到半径的标定。背景是有明确位置的代表混合谱，类别名称用于教学说明。

`representative_scan.npz` 保存五个固定半径对四条旧控制谱的完整解，包括 `fractions`、`normalized`、`shade`、`rms`、`condition`、`qa`、`modeled`、`residual`。后三条背景端元可由字典自身恢复，所以近零雪系数和残差属于自拟合控制，不能据此给像元标注独立无雪真值。`E_c = observed − modeled`，RMS 为逐波段残差的均方根。

`fixed_radius_summary.csv` 按半径汇总共同像元上的系数、比例和光谱 RMS；`image_endmembers.csv` 列出五条雪谱的分位、位置和吸收面积；`image_endmember_summary.csv` 比较相同背景下的物理雪库与影像雪谱。`spatial_metadata.json` 给出 CRS、仿射变换、形状、选样规则及正演太阳角设定。

## MODIS 原生 1 km 单元

`modis_1km_units.csv` 保留 420 条 ROI 内记录，键为 `roi + tile_row_1km + tile_column_1km`。两个 ROI 在边缘重复涉及 14 个原生单元。全区统计先按原生行列去重，得到 406 个单元；同一 ROI 内重复或重叠记录的状态标志冲突会报错。重叠单元保留有效支持较完整的一条，面积并列时按 ROI 名称确定顺序。`flag12` 为 MOD35 snow/ice，`flag15` 为 internal snow mask。`aviris_fsca` 与 `modis_fsca` 分别为 2013 年影像背景 FCLS 模型聚合后的雪比例。`*_valid_area_fraction` 为有效模型输出覆盖目标单元的面积比例；`common_valid` 要求两者均达到 95%。复制到 500 m 的标志不增加独立统计单元。

`product_consistency.csv` 按区域、模型、参照标志和阈值（0.1、0.5、0.9）列出 TP、FN、FP、TN、参照召回率和特异度。缺少某类时对应比率为空，`n=0` 表示严格共同掩膜内没有可比单元。这些指标描述产品一致性。

## 已归档的独立验证与尺度统计

`*_corrected_coverage.csv` 与 `same_dictionary_control_corrected.csv` 为已有面积支持规则修正后的尺度结果。两份 `*_sample_pixels.csv.gz` 与 `stratified_metrics.csv` 为 Landsat/WorldView 两场独立覆盖参考对照，参考来源为 Stillinger 和 Bair 的 CC BY 4.0 数据集（DOI: 10.5281/zenodo.4031446）。这些文件属于 `legacy_library_output`：沿用原报告的历史模型结果，与本轮采用影像背景的新实验分别展示。

WorldView 表的 `reference_fsca` 为参考雪覆盖比例，`stratum` 划分无雪、混合雪和近纯雪；方法列为模型雪比例。`common_four_methods_normalization_valid` 指四种方法共同具有有效归一化比例，统计时必须应用同一掩膜。2016 配对为同日，2017 配对相差一天。

两份样本表采用 `gzip.compress(..., compresslevel=6, mtime=0)` 无损压缩，全部 12,000 行、44 列分别保留。共 7,344,001 B 压缩为 1,881,183 B；压缩文件与解压原文 SHA-256 同时保存在 manifest 中。`pandas.read_csv` 自动识别 gzip，读取接口拒绝同一场景 CSV/GZ 并存，避免重复计数。原始计算的 `code_sha256` 保留生成时来源，第二版仅新增输入提取与无损打包记录。

同一原生 1 km 集合的筛选账本为 406 候选→406 有效状态→100 AVIRIS 有效面积≥95%→17 共同有效面积≥95%。bit 12 阳性为 31→31→0→0，bit 15 为 116→116→20→0。`state_flag_qa_ledger.csv` 再分解面积合格后的状态 QA：100 个单元（20 个 bit 15 阳性）→55 个晴空单元（10 个阳性）→17 个无云邻接单元（0 个阳性）。17 个阴性单元仍保留；85 个历史 500 m 配对独立统计，不接入该漏斗。

## 固定粒径的归一化诊断与空间不确定度

`fixed_radius_uncertainty.csv` 每行一个半径，`common_n=60949`。`denominator_mean/p05/p50`均在五模型共同域计算；`denominator_pass_fraction_common`在该域均为1，`denominator_pass_fraction_input`以全部62908个输入为分母。分母从重新求解的原始系数直接求和，不以`raw_snow/normalized_snow`反推。无约束负值保留，`negative_snow_fraction`给出其比例。

`normalized_snow_ci_low/high`与`rms_ci_low/high`为像元加权均值的95%区间，单位分别为比例与反射率。`fixed_radius_bootstrap.json`记录ROI左上角为原点、50×50像元（1.5 km）、30个有效区块、1000次、seed=1998及1000−50 μm的配对差值。各模型每次抽同一组区块，边缘块按像元数加权，区间描述固定场景与端元下的空间抽样。

`coverage_sensitivity.csv`每档面积阈值保留AVIRIS支持、共同支持两行，比较0.80、0.90、0.95；完整候选键来自`modis_1km_units.csv`，不与历史85个500 m配对混合。

`sample_support.csv`集中列出各分支的30 m像元口径与父集合。63213、62908、60949、62861、62870分别来自历史ROI、新影像背景输入、五固定半径共同域、三方法共同域、历史仅光谱退化共同域；末三项有各自掩膜，不构成连续筛选链。

`lab_conditions.csv`列出Hannula与Heinilä（Zenodo 3580078，CC BY 4.0）的10个实验室雪况及日期、干雪标记与条件级SSA代理，已从原始实验表选列。覆盖指标使用6个干雪条件，半径使用其中4个有SSA的条件；条件标识便于追溯，不能把重复光谱计为独立雪况。

## 有符号系数与留出光谱

`coefficient_diagnostics.npz` 的 `fractions` 轴为 `(3 methods, 62908 pixels, 4 endmembers)`；端元顺序为雪、植被优势、亮裸地、暗裸地。`shade`、`rms`、`qa` 均为 `(3,62908)`。方法顺序由 `method_names` 给出；系数为无量纲 float32，RMS 为反射率单位。去阴影比例从原始系数之和重新计算。`valid_mask`、`ndsi`、`ndvi` 形状均为 `(267,267)`，像元顺序是掩膜的 C-order 展平顺序；`transform` 为六参数仿射变换，CRS 见 `spatial_metadata.json`。

`coefficient_feasibility.csv` 区分各自归一化域与三方法共同域；`coefficient_spectral_groups.csv` 在共同域按 NDSI/NDVI 分层，记录负雪系数数量。物理可行性同时检查四个地物系数、阴影和闭合，存储精度容差为 10⁻⁷。图中有符号雪系数与越界类别分别显示。

`heldout_spectra.npz` 保留四条 `(4,182)` 观测、零起算 `row_column`、NDSI/NDVI，以及五个半径下的原始解、重建和残差。选样排除所有背景与候选雪端元周围三像元，预先按指数分层，不依据解混误差选择。它们是谱型教学样本，没有独立覆盖标签。

## 已知真值的模型失配

`mismatch_repeats.csv` 为 7 条件×3 方法×20 重复的 420 行，`mismatch_summary.csv` 为 21 行汇总。每个条件每次有 48 条光谱：4 半径×6 雪比例×2 阴影。方法共享同一批观测，seed=1998；变量、波段及失配构造见 `snowmix/validation_study.py`。覆盖误差与光谱 RMS 均以百分点输出，半径误差为 μm。匹配库、两档噪声、半步网格半径、未入库松针背景、太阳角失配、平滑谱扰动分别比较。这里的区间来自重复实验，不是独立场景区间。

## 公开端元的 Landsat 转移实验

`lodgepole_pine.asc` 为 USGS splib05a 原始公开松针文件；`transfer_library.source.json` 保留作者、来源、哈希及波段卷积说明。`transfer_library.npz` 的 `snow` 为 `(2 scenes,108 radii,6 Landsat bands)`，`backgrounds` 为 `(6,2)`，对应松针与石英；`angles_deg` 是两景角度。`unseen_background_aviris` 和 `mismatch_band_indices` 为失配实验提供 182 个有充分支持的波段。

`transfer_metrics.csv` 从两份完整 LC08 样本表重新解混，以各方法共同归一化域评价；`transfer_paired_deltas.csv` 为多粒径减固定半径的配对 RMSE 差及 1.5 km 区块重采样区间，负值表示多粒径误差较小。此分支有六/四波段、MGS/FCLS，与历史花岗岩分支分别命名。`historical_paired_deltas.csv` 只重算历史保存预测的配对统计；`historical_spectral_control.npz` 保存历史仅光谱退化的共同域结果，用于补齐尺度对照四行。`experiment_scope.csv` 汇总各分支数据、方法和评价性质。

## 新增同日 MODIS 独立覆盖验证

原始 MOD09GA 为 2016-02-13 和 2017-05-21 的 h08v05、Collection 6.1。产品标识、HDF SHA-256、原生网格、比例因子、太阳角及公开参考哈希见 `modis_external.source.json`。参考采用 Stillinger 与 Bair 在 Zenodo 4031446 以 CC BY 4.0 公开的雪覆盖派生产品，已作裁剪、相交面积聚合和分层；不包含原始商业影像。

| 文件 | 字段与单位 |
|---|---|
| `modis_reference_pixels.csv.gz` | 参考范围内的原生 500 m 行列；`B1`–`B7` 为已缩放反射率，`reference_fsca` 为 0–1 雪比例，`reference_coverage` 为有效参考面积/目标面积；QA 与原始状态整数、太阳天顶角（°）同时保存 |
| `modis_reference_1km.csv` | 原生 1 km 行列、面积加权参考雪比例及支持面积，`state_clear_no_adjacent`、bit 12 和 bit 15；评价不额外要求七波段反射率都通过 QA |
| `modis_external_library.npz` | `snow` 为 `(2,108,7)`，由 NumPy WW80 按各景代表太阳角计算；`backgrounds` 为 `(7,2)`，USGS 松针和石英 |
| `modis_external_support.csv` | 每景、每 QA 分支的参考支持、QA、归一化排除、四方法共同域及雪分层数量 |
| `modis_external_metrics.csv` | 每景×QA 分支×方法×分层；覆盖 bias/RMSE 与光谱 RMS 单位均为 pp，越界率与可行率为 0–1 |
| `modis_external_paired_deltas.csv` | 多粒径减固定半径的 RMSE 差及空间区块配对区间，pp |
| `modis_external_flags.csv` | 1 km 标志 TP/FN/FP/TN、召回率和特异度；参考阈值为 0.1、0.5、0.9 |

`strict7` 保留原有严格七波段 QA；`selected5` 是预先规定的 1/2/4/6/7 波段质量对照。每个分支四种模型采用同一共同归一化域。基本分层为无雪 `<0.1`、混合雪 `[0.1,0.9)`、近纯雪 `≥0.9`；补充的互斥三层为无雪、低覆盖混合 `[0.1,0.5)`、雪占优 `≥0.5`，雪占优不等同纯雪。2016 年 strict7 共同域 1,416（463 个参考阳性），2017 年 1,817（22 个阳性）；独立 1 km 状态域分别为 408（123 阳性）与 449（5 阳性）。

所有 CSV 日期标识按字符串读取，例如 `dtype={"scene": str}`。三份 gzip 均为确定性无损压缩；manifest 同时记录压缩字节与解压内容哈希。
