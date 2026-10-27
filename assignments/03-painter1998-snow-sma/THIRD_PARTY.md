# 第三方来源与许可证

| 内容 | 冻结版本 | 授权及保留文件 |
|---|---|---|
| SNICAR-ADv3 | `227e9de9b100ed0671662f94d2040fd8231aab91` | MIT；`vendor/SNICARv3/LICENSE` |
| SPIRES 双流及尺寸函数 | `bcca68ebad36a34bf67f8bc88b7198eda9fc608f` | MIT；`vendor/SPIRES/LICENSE` |
| Python 运行依赖 | `requirements-lock.txt` | 安装时保留各项目自身许可；附件只列版本 |
| 三组 Zenodo 验证资料 | 3580078、3580825、4031446 | CC BY 4.0；元信息保留 DOI 和来源 |
| USGS/NASA/JPL 数据 | 来源表具体样本及产品 | 按相应产品说明引用原数据和传感器机构 |
| nbconvert / Jupyter HTML 模板 | 7.16.6 | BSD-3-Clause；仓库 `site/assets/nbconvert/LICENSE（仓库根）` |
| MathJax SVG 公式渲染 | 3.2.2 | Apache-2.0；仓库保留来源锁与许可，组件经固定校验和下载到忽略缓存；HTML旁保留完整资产及许可 |

SNICAR 版权：Copyright (c) 2020 Mark Flanner and SNICAR contributors。
SPIRES 版权：Copyright (c) 2023 Jeff Dozier。

两份完整 MIT 许可证随源码保留。使用的 `twostream.m` 来自 SPIRES `SolarLongwaveRadiation/`，`checkSizes.m` 来自 `General/`。对应冻结原件的 URL 和 SHA-256 位于 `data/sources/`。

代码来源：[SNICAR](https://github.com/mflanner/SNICARv3)、[SPIRES](https://github.com/edwardbair/SPIRES)。SNICAR 工作副本由 `prepare_optics.py` 生成，仅修改光学目录；vendor 冻结源保持原样。

实验室谱资料由 Hannula, Henna-Reetta 和 Heinilä, Kirsikka 发布；野外谱资料由 Heinilä, Kirsikka、Böttcher, Kristin 和 Mattila, Olli-Pekka 发布；WorldView 验证集由 Stillinger, Timbo 和 Bair, Ned 发布。论文及数据集的完整引用以课程报告参考文献为准，下载资料时沿用各记录的引用信息。

新增松针使用 Clark, R. N., Swayze, G. A., Wise, R. K., et al. (2003), *USGS Digital Spectral Library splib05a*, USGS Open-File Report 2003-395，[DOI: 10.3133/ofr03395](https://doi.org/10.3133/ofr03395)。`lodgepole_pine.asc` 保留公开原文；随后仅在有效波长支持内进行传感器响应卷积。石英同属该公开库。原始文件与派生数组分别记录哈希。

2016-02-13、2017-05-21 的 MOD09GA.061 为 NASA Terra 地表反射率公开产品。HDF 身份与元数据分别见 [2016 场景](https://cmr.earthdata.nasa.gov/search/concepts/G2263526666-LPCLOUD.html)、[2017 场景](https://cmr.earthdata.nasa.gov/search/concepts/G2262531885-LPCLOUD.html)。教学表已应用比例因子、筛选空间范围、提取质量位并聚合公开覆盖参考；不再分发整景 HDF。

Stillinger, T. 与 Bair, N. 的[雪覆盖验证派生资料](https://doi.org/10.5281/zenodo.4031446)采用 CC BY 4.0。新增 MODIS 分支使用该资料的公开雪覆盖产品，实施相交面积加权和阈值分层；原始商业 WorldView 不在案例内。这里的修改与派生结果由本项目负责，不表示原数据作者认可本项目结论。

HTML 的模板样式来自 [nbconvert](https://github.com/jupyter/nbconvert/tree/7.16.6)，本地公式组件来自 [MathJax](https://www.npmjs.com/package/mathjax/v/3.2.2)。字体使用本机已安装字体，不分发微软字体。公开数据及排除项另见 `PUBLIC_DATA_POLICY.md`。

原创代码的 MIT 适用范围在 README 中说明；报告和原创教学文字单独采用 CC BY 4.0。外部模型输入和数据不会因收入本仓库而改变许可。

`data/demo/scattering_inputs.npz` 从 SNICAR-ADv3 官方公开 202105 NetCDF 光学库提取两个冰选项、148 个半径；`ss_alb` 保留 float64，`asm_prm` 转为 float32，波长从 m 转为 μm。这是输入属性提取与精度转换，没有重新计算 Mie 属性。作者为 Mark G. Flanner 与 SNICAR contributors；模型论文 DOI: 10.5194/gmd-14-7673-2021 的 Code and data availability 提供代码及配套库，结论中明确面向研究和教育。源库和每个提取文件哈希保存在同名 `.source.json`；保留外部来源说明，不将库数据重新声明为项目 MIT。
