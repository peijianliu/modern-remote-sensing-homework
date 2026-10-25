# 数据来源与读取约定

| 输入 | 场景、样本或版本 | 获取及用途 |
|---|---|---|
| AVIRIS-Classic L2 | `f130503t01p00r14_sc01`，2013-05-03 | JPL 反射率包，14.3 m 原生网格，面积聚合至 30 m |
| MOD09GA C6.1 | `MOD09GA.A2013123.h08v05.061.2021229143331` | 同日 Terra h08v05；Earthdata 下载，HDF-EOS 导出 |
| SNICAR 光学属性 | 202105，无藻类包 | 官方 480 波段属性，球形雪与两种冰折射率选项 |
| USGS 库谱 | Quartz HS32、lodgepole pine | 石英用于合成实验；松针用于历史库谱基准 |
| ECOSTRESS 历史库谱 | Granite H2、Cryumbrept 87P3855 | 历史基准来源；新增计算采用影像背景，原谱不随仓库分发 |
| Terra RSR | MCST `Terra_RSR_in-band.xlsx` | 7 波段归一化 channel 响应平均 |
| 实验室雪谱/SSA | Zenodo 3580078 | 10 雪况、158 重复、7 组 SSA |
| 野外目标谱 | Zenodo 3580825 | 87 个目标，按 ID 匹配谱及样本信息 |
| WorldView 雪覆盖参考 | Zenodo 4031446 | 可见雪 fSCA；选择 2016、2017 两场 |
| Landsat C2 L2 | `LC08_L2SP_043033_20160213_02_T1`、`LC08_L2SP_043032_20170522_02_T1` | 按 WorldView 范围取得原生窗口 |
| Landsat OLI RSR | USGS Spectral Characteristics Viewer | 6/4 波段迁移检验 |

主数据源清单位于 `data/sources/*.json`；公开验证的元信息汇总位于 `metadata/public_validation_sources.json`。清单保留公开链接、文件字节数和 SHA-256；许可及引用见公开记录和 `THIRD_PARTY.md`。随仓库提供的 `data/demo/aviris_metadata.json` 包含波段及投影元信息；`prepare_test_inputs.py` 可将它复制到历史测试所用的 `data/cache/` 路径。

公开入口：

- AVIRIS：[JPL 门户](https://aviris.jpl.nasa.gov/dataportal/)，[本场 L2 包](https://popo.jpl.nasa.gov/avcl/y13_data/f130503t01p00r14_refl.tar.gz)。
- MOD09GA：[同日场景](https://cmr.earthdata.nasa.gov/search/concepts/G2261932874-LPCLOUD.html)。
- SNICAR：[官方代码](https://github.com/mflanner/SNICARv3)，[202105 光学属性](http://snow.engin.umich.edu/opticalprops/snicar_v3_optics_202105/snicar_v3_optics.tar.gz)。
- 谱库：[USGS](https://www.usgs.gov/data/usgs-spectral-library-version-7-data)；[ECOSTRESS](https://speclib.jpl.nasa.gov/library) 仅列为历史实验来源。新增影像背景的选取规则与位置见 `data/demo/README.md`。
- 验证资料：[实验室](https://zenodo.org/records/3580078)，[野外](https://zenodo.org/records/3580825)，[WorldView](https://zenodo.org/records/4031446)。
- Landsat：[Microsoft Planetary Computer 公开目录](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)，[USGS 光谱响应](https://www.usgs.gov/landsat-missions/spectral-characteristics-viewer)。

## 输入与生成路径

`data/sources/*.json` 和 `metadata/public_validation_sources.json` 中的文件路径均相对于本作业目录。默认 Notebook 读取仓库内 `data/demo/`。完整处理另使用 `data/raw/` 原始资料和 `data/cache/` 解压影像、传感器窗口及正演缓存。

`prepare_case_data.py --source-root` 面向已完成基础处理的实验工作区，除原始影像、MODIS 导出和模型缓存外，还读取已归档的 v2 尺度及 WorldView 统计目录。相关历史 CSV 已冻结在 `data/demo/`，默认教学计算直接重用这些文件；完整数据包的重建需要同时保留这些归档统计，不能只准备一场原始影像。

MOD09GA 需通过 Earthdata 登录下载，将完整 HDF 文件保存为 `data/raw/MOD09GA.A2013123.h08v05.061.2021229143331.hdf`。来源清单的 SHA-256 对应所用原始文件，便于核对下载版本。

## 单位与质量筛选

波长、FWHM 和光学等效半径为 μm；反射率和比例为无量纲。历史 ECOSTRESS 数据按 percent 除以 100；AVIRIS 存储值除以 10000，MOD09GA 反射率乘 0.0001。MOD09GA reflectance fill 为 −28672、QC fill 为 787410671；云、阴影、卷云、邻云和坏波段按位筛选，雪 QA 位保留。

SSA 为 m²/kg，以冰密度 917 kg/m³ 换算 `r=3e6/(917×SSA)`；最高高度样本作为条件级近表层参照。Landsat SR 按 `DN×0.0000275−0.2` 只缩放一次；QA 无缩放。WorldView fSCA 按 Int16/1000 读取，−32768 为空值。2016 配对为同日，2017 相隔 1 d，统计按场景分别计算。

阴影归一化系数保存为 `normalized`，原始系数为 `fractions`；`E_c` 和 RMS 使用原始系数。QA 位：1 输入无效、2 无满秩模型、4 分母不足、8 低雪、16 库边界、32 条件数偏高、64 系数越界。真值、合成比例和另一传感器模型参照按实验定义分别记录。
