# 综合报告图表与代码索引

对应33页综合报告的14幅编号图。Notebook 直接使用随仓库提供的教学数据；历史完整流程的绘图脚本需先按数据说明准备输入及归档结果。

| 图号 | 内容 | 重绘或计算入口 | 输入范围 |
|---|---|---|---|
| 1 | 航带和研究区位置 | `scripts/course_report_assets.py` | 完整 AVIRIS 元信息与研究区配置 |
| 2 | 0°/60°纯雪光谱 | Notebook 01 | 冻结正演缓存 |
| 3 | 半径—反照率 | Notebook 01 | 冻结正演缓存 |
| 4 | Fig. 3 误差面与纯雪覆盖 | Notebook 02 | 缓存、石英谱及波段信息，重新解混 |
| 5 | 历史库谱 FCLS 系数 | `scripts/course_report_assets.py` | 历史 AVIRIS 解混图层 |
| 6 | 粒径与质量标记 | `scripts/publication_figures.py` | 历史图层；Notebook 03 提供新增分支图层 |
| 7 | 观测、重建与残差 | `scripts/publication_figures.py` | 历史代表像元；Notebook 03 重算新增代表像元 |
| 8 | 五个固定半径扫描 | Notebook 03、`snowmix/case_analysis.py` | 共同归一化集合60,949，背景与波段保持一致 |
| 9 | 五种影像雪谱 | Notebook 03、`snowmix/case_analysis.py` | 吸收面积分位选谱及共同观测集合 |
| 10 | 留出波段检验 | `scripts/aviris_band_sensitivity.py` | 完整 AVIRIS 输入及预定样本 |
| 11 | 固定字典尺度对照 | `scripts/resolution_control.py`；Notebook 03 重算表格 | 保存的修正面积支持统计 |
| 12 | 严格 QA 双源散点 | `scripts/publication_figures.py`；Notebook 03 重算配对统计 | 两区 corrected_coverage 表 |
| 13 | 实测雪谱与 SSA 代理 | `scripts/empirical_validation.py` | Zenodo 公开实测输入与特定角度缓存 |
| 14 | WorldView 分层结果 | `scripts/worldview_validation.py`；Notebook 03 重算分层表 | 两场公开参考的派生样本表 |

新版报告的表4区分全网格平均偏差与两个纯雪端点偏差；固定半径表图统一使用共同归一化域。MODIS雪标志交叉表采用唯一1 km单元，完整数据见 `data/demo/product_consistency.csv`。历史库谱与新增影像背景结果按分支分别解释。

Notebook 与脚本的完整对应见 `CODE_MAP.md`；字段、单位、CRS和数据来源见 `data/demo/README.md` 与 `manifest.json`。
