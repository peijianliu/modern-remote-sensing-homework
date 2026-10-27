# 数据来源与读取约定

[数据许可](PUBLIC_DATA_POLICY.md) · [教学字段](data/demo/README.md) · [实验分支](EXPERIMENTS.md)

| 输入 | 产品、样本或版本 | 用途 |
|---|---|---|
| AVIRIS-Classic L2 | 2013-05-03，`f130503t01p00r14_sc01` | 原生14.3 m，面积聚合至30 m |
| MOD09GA C6.1 | 2013-05-03，Terra h08v05 | 原实验同日双源和状态对照 |
| MOD09GA C6.1扩展 | 2016-02-13、2017-05-21，h08v05 | 与同日公开覆盖参考独立对照 |
| SNICAR光学属性 | 202105，480波段，球形雪与两套冰选项 | 提取散射输入，NumPy WW80重算 |
| USGS splib05a | Quartz HS32，Lodgepole Pine LP-Needles-1 | 合成、公开字典与未入库背景对照 |
| Terra RSR | MCST `Terra_RSR_in-band.xlsx` | 七波段响应积分 |
| 实验室雪谱/SSA | Zenodo 3580078，CC BY 4.0 | 条件级光谱与SSA对照 |
| 野外目标谱 | Zenodo 3580825，CC BY 4.0 | 87目标，按ID匹配 |
| WorldView派生参考 | Zenodo 4031446，CC BY 4.0 | 可见雪覆盖；30 m派生产品再聚合 |
| Landsat C2 L2 | 043033/20160213、043032/20170522 | 两景原生B2–B7窗口及保留样本 |
| OLI RSR | USGS Spectral Characteristics Viewer | 六/四波段转移实验 |

ECOSTRESS Granite H2、Cryumbrept 87P3855仅列作历史库谱统计的来源。其原谱不进入新实验，下载与读取入口已排除这两条记录。商业原始WorldView不获取、不分发；采用的是作者明确授权的派生雪覆盖产品。

## 官方入口与版本

- AVIRIS：[JPL门户](https://aviris.jpl.nasa.gov/dataportal/)、[本场L2包](https://popo.jpl.nasa.gov/avcl/y13_data/f130503t01p00r14_refl.tar.gz)。
- MOD09GA：[2013场景](https://cmr.earthdata.nasa.gov/search/concepts/G2261932874-LPCLOUD.html)、[2016场景](https://cmr.earthdata.nasa.gov/search/concepts/G2263526666-LPCLOUD.html)、[2017场景](https://cmr.earthdata.nasa.gov/search/concepts/G2262531885-LPCLOUD.html)。
- SNICAR：[官方代码](https://github.com/mflanner/SNICARv3)、[光学库入口说明](https://github.com/mflanner/SNICARv3/blob/master/Links%20to%20optics%20libraries.md)。采用202105归档，按`data/sources/optics.json`核对完整归档SHA-256。
- 光谱：[USGS splib05a及正式引用](https://doi.org/10.3133/ofr03395)；原始石英与松针文件保留头信息。
- 验证资料：[实验室](https://zenodo.org/records/3580078)、[野外](https://zenodo.org/records/3580825)、[可见雪覆盖](https://zenodo.org/records/4031446)。
- Landsat：[公开目录](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)、[USGS响应](https://www.usgs.gov/landsat-missions/spectral-characteristics-viewer)。

原有来源清单在`data/sources/`和`metadata/public_validation_sources.json`。新MODIS的完整产品ID、原始HDF哈希、派生参考哈希、实际格网及角度范围在`data/demo/modis_external.source.json`。NASA数据服务可能要求Earthdata登录；账户和临时链接不进入仓库。

## 完整公开输入目录

L2的`--source-root`指向已准备的公开输入目录，文件路径相对于该根目录：

```text
data/cache/aviris_metadata.json
data/cache/aviris/f130503t01p00r14_refl/..._refl_img_corr
data/cache/aviris/f130503t01p00r14_refl/..._refl_img_corr.hdr
data/raw/Terra_RSR_in-band.xlsx
data/raw/public_validation/zenodo_4031446/
    16FEB13190431_landsat_30m_fsca.tif
    17MAY21191849_landsat_30m_fsca.tif
```

AVIRIS元数据中的`binary_path`定位完整ENVI；`prepare_aviris.py`可从原始归档生成这些文件。现行重建入口使用仓库内已公开的雪谱作为端元输入，从完整影像重新选取背景和五条影像雪谱，冻结的预测结果只用于事后校核。

独立MODIS还需两份原始HDF，经`prepare_modis_hdf.py`在新的`<field-root>/modis_2016/`、`modis_2017/`中导出`modis_fields.mat`、`modis_attributes.mat`和原生产品核对记录。再运行`prepare_modis_validation.py`。命令见README。

`prepare_case_data.py`、`python -m snowmix`及其他历史脚本保留原归档的处理路线；它们还依赖历史模型缓存和统计，不能代替现行L2入口。历史原谱排除规则同样适用，不静默更换原字典后沿用旧精度数字。

## 光学参数与参考重建

从官方202105归档解压后，可重建小型散射参数包：

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_teaching_optics.py --prepare-optics --optics-root <optics-root>/snicar_480band --output build/scattering_inputs.npz
& .\.venv\Scripts\python.exe scripts/prepare_forward_reference.py --prepare-reference --workspace <prepared-workspace> --matlab <matlab-executable>
```

`ss_alb`以float64、`asm_prm`以float32保存，计算提升至float64。MATLAB使用显式34字段输入，完整MAT只作可选校核及历史输入；默认Notebook不依赖它。小型SNICAR光谱库是已标来源的模型端元输入，与可直接重算的WW80路径分别说明。

## 单位 质量与统计对象

波长、FWHM与半径用μm，反射率、反照率、原始系数和去阴影比例均无量纲。AVIRIS uint16除10000；MOD09GA int16乘0.0001，填充值−28672，QC填充值787410671。太阳角原始值乘0.01转换为度。比例因子从实际HDF属性核对。

Landsat为`DN×0.0000275−0.2`，只缩放一次；QA无缩放。公开WorldView派生覆盖按Int16/1000，显式屏蔽−32768。30 m参考按真实相交面积聚合，部分覆盖的有效面积不补零为无雪。1 km参考在再次聚合时按500 m有效支持面积加权。

原始`fractions`重建反射率，`E_c=observed-modeled`。去阴影比例`normalized`只在分母>0.05时输出。QA位为1输入无效、2无满秩模型、4分母不足、8低雪、16库边界、32条件数、64系数越界。

2013两个ROI有420条记录、406个原生1 km单元，重叠处按有效支持择取。独立反射率误差按500 m单元，状态标志按唯一1 km单元统计；不同日期与不同字典分别报告。

SSA单位m²/kg，按`r=3e6/(917×SSA)`转为μm，最高高度记录作为条件级近表层代理。2016 Landsat与参考同日，2017相隔1 d；新增MOD09GA两场与参考均同日。评价对象和局限见综合报告及EXPERIMENTS。
