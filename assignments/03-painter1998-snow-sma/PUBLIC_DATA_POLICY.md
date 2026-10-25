# 公开数据与使用条件

案例使用公开遥感产品、公开光谱和具有明确许可的验证资料。来源链接、样本编号、单位、处理方式及文件哈希与教学数据一并记录。需要 Earthdata 账户下载的 NASA 开放产品仍属于公开数据；账户信息不进入仓库。

| 数据 | 公开与使用依据 | 本仓库提供的内容 |
|---|---|---|
| 2013 AVIRIS-Classic L2 | JPL 公开数据门户；NASA Earth Science 数据政策 | 小范围光谱、选谱位置和派生图层；完整航带由原门户获取 |
| MOD09GA C6.1 | NASA 开放产品；保留产品编号与 NASA 引用 | 研究区质量位、唯一 1 km 单元表和解混派生统计 |
| USGS 光谱 | USGS 公开数字光谱库及 public-domain 使用政策 | Quartz HS32 原谱、波段卷积与引用信息 |
| SNICAR / WW80 正演 | SNICAR、SPIRES 的 MIT 源码 | 本项目计算的冻结正演结果；不打包完整光学属性数据库 |
| 实验室与野外雪谱 | Zenodo 3580078、3580825，Open、CC BY 4.0 | 原有实测验证的统计和引用信息 |
| WorldView 可见雪覆盖参考 | Zenodo 4031446，Open、CC BY 4.0 | 作者公开发布的派生雪覆盖参考及本项目分层统计 |
| Landsat C2 L2 | USGS 开放产品 | 公开验证的派生像元表，不捆绑整景数据 |

## 未纳入的材料

- 原始商业 WorldView 影像不随案例获取或分发；使用的是作者在 Zenodo 公开授权的雪覆盖派生数据。
- EcoSIS 城市砾石候选的单位与再分发许可未确认，本版不使用。
- ECOSTRESS 中的 JHU Granite H2、87P3855 原谱可公开访问，但原谱再分发条款尚不明确。本版新增计算及教学背景改为公开 AVIRIS 影像选谱，不分发这两条原谱或完整重采样谱。报告中的历史库谱实验统计保留原始来源引用。
- 五份组员原始报告不公开；报告中的不同场景数字不作为本案例计算输入。
- 登录凭据、临时授权链接和个人目录不进入数据说明、Notebook 或 HTML。

## 来源与引用

- [NASA 数据使用与引用政策](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy)
- [AVIRIS 数据处理与公开分发](https://aviris.jpl.nasa.gov/aviris/data_facility.html)
- [USGS 数字光谱库 splib05a](https://pubs.usgs.gov/of/2003/ofr-03-395/)
- [USGS 数据使用政策](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)
- [实验室雪谱与 SSA](https://doi.org/10.5281/zenodo.3580078)
- [野外雪与裸地光谱](https://doi.org/10.5281/zenodo.3580825)
- [可见雪覆盖验证掩膜](https://doi.org/10.5281/zenodo.4031446)
- [ECOSTRESS 谱库说明与署名要求](https://speclib.jpl.nasa.gov/)

核对日期：2026-10-24。外部数据保留原许可，仓库的 MIT 代码许可不覆盖这些数据。
