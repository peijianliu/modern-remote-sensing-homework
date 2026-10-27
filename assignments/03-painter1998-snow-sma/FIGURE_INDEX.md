# 综合报告图表与代码索引

对应 2026-10-27 教学验证修订版的 19 幅编号图。默认 Notebook 从教学输入重算；注明“历史”的图沿用该分支保存的结果，完整入口需要另外准备原始输入和归档结果。

| 图号 | 内容与分支 | 重绘或计算入口 | 输入与评价对象 |
|---|---|---|---|
| 1 | 2013 航带及研究区 | `scripts/course_report_assets.py` | AVIRIS 元信息、两个 ROI；包络不代表全部有效覆盖 |
| 2 | 0°/60°纯雪光谱 | Notebook 01；`scripts/report_diagnostics.py` | 官方散射属性，NumPy WW80；方向–半球反照率 |
| 3 | 半径与加权反照率 | Notebook 01；`scripts/report_diagnostics.py` | 相同散射输入与太阳谱；W84/WB08 对照 |
| 4 | Painter Fig. 3 误差面和纯雪覆盖 | Notebook 02；`scripts/publication_figures.py` | 146×101 条已知真值光谱；模型内恢复 |
| 5 | 七种模型失配 | Notebook 02；`scripts/report_extended_figures.py` | 182 波段受控混合，三方法共享观测，20 次重复 |
| 6 | 历史库谱 FCLS 原始系数 | `scripts/course_report_assets.py` | 历史 AVIRIS 图层，含零谱阴影，未去阴影归一化 |
| 7 | 历史候选半径与 QA | `scripts/publication_figures.py` | 历史模型图层；按库边界、低雪、小分母分类 |
| 8 | 历史观测、重建与 E_c | `scripts/publication_figures.py` | 原始系数重建，保留近吸收带残差 |
| 9 | 当前系数符号与可行性 | Notebook 03；`scripts/report_extended_figures.py` | 同一 2013 影像背景、62,861 共同像元；MGS 与 FCLS |
| 10 | 固定半径及空间块区间 | Notebook 03；`scripts/report_diagnostics.py` | 五模型共同域 60,949；分母和配对区间 |
| 11 | 五种影像雪谱 | Notebook 03；`snowmix/case_analysis.py` | 预定吸收面积分位选样；吸收面积不转换半径 |
| 12 | 历史公共留出波段 | `scripts/aviris_band_sensitivity.py` | 1,200 个预定随机像元；相同 30 个未拟合波段 |
| 13 | 历史固定字典尺度对照 | `scripts/resolution_control.py`；Notebook 03 重算表格 | 修正面积支持；四种尺度/处理顺序对照 |
| 14 | 历史严格 QA 双源散点 | `scripts/publication_figures.py`；Notebook 03 | 两 ROI 的 85 个 500 m 配对；模型一致性 |
| 15 | 原生 1 km 筛选与阳性类别 | Notebook 03；`scripts/report_validation_figure.py` | 420 条 ROI 记录去重为 406 单元，406→100→17；阈值表另列 80/90/95% |
| 16 | 2013 支持范围与 QA 空间图 | Notebook 03；`scripts/report_extended_figures.py` | 原生正弦网格，候选、面积、标志及首个排除条件 |
| 17 | 实测雪谱与 SSA 代理 | `scripts/empirical_validation.py` | Zenodo 公开测量；按雪况计数，条件标识见表注 |
| 18 | 历史 WorldView 分层散点 | `scripts/worldview_validation.py`；Notebook 03 重算指标 | 历史 FCLS 库谱，2016 公开派生覆盖；图为各方法域，表为共同域 |
| 19 | 新增同日 MODIS 独立覆盖误差 | Notebook 03；`scripts/report_extended_figures.py` | 2016/2017，WW80 与公开 USGS 背景；四方法共同域，总体/混合雪分开 |

报告表 5 区分全合成网格 bias 与纯雪端点 bias；表 7 集中说明各实景分支的计数；表 8 报告共同域物理可行性；表 9 给固定半径分母与区块区间；表 14 给面积门槛敏感性；表 19–22 对应新增公开端元 Landsat 与 MODIS 的独立覆盖和雪标志检验。独立参考来源和修改说明见 `DATA_SOURCES.md`。

教学补图 `reports/figures/public_transfer_validation.*` 对应 Notebook 03 和首页的公开端元 Landsat 混合雪比较，不另占报告图号。完整字段与单位见 `data/demo/README.md`，模块与命令见 `CODE_MAP.md`。
