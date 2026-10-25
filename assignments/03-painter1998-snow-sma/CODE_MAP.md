# 代码与图表索引

| 方法或结果 | 实现 | Notebook |
|---|---|---|
| MGS、FCLS、逐候选最小 RMS、质量标记 | `snowmix/linear.py` | 02、03 |
| 库谱读取、单位转换、高斯响应卷积 | `snowmix/spectra.py` | 01、02 |
| 物理缓存解释、Fig. 3 重算与教学绘图 | `snowmix/teaching.py` | 01、02、03 |
| 吸收面积、分位选谱、固定半径扫描、交叉表 | `snowmix/case_analysis.py` | 03 |
| 教学数据及新实验准备 | `scripts/prepare_case_data.py` | 完整输入准备后运行 |
| 真实像元面积聚合与 MODIS QA | `snowmix/geo.py`、`modis.py`、`image.py` | 03 使用其冻结派生输入 |
| 执行与离线 HTML 导出 | `scripts/build_notebooks.py` | 三份 Notebook |
| MATLAB 正演参考 | `matlab/run_forward.m`、`vendor/` | 01 读取其计算缓存 |

报告中的历史物理图、Fig. 3、噪声/端元遗漏和外部验证来自已归档实验；本版新图来自教学数据中的统一 2013 AVIRIS 背景实验。图表号以综合报告 PDF 为准，数据字段与来源可由 `data/demo/README.md` 和 `manifest.json` 核对。
