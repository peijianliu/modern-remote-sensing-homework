# 代码与图表索引

| 内容 | 实现 | 教学入口 |
|---|---|---|
| MGS、FCLS、候选雪谱与残差/QA | `snowmix/linear.py` | 02、03 |
| WW80闭式解、波段响应与单位 | `optics.py`、`spectra.py` | 01、02 |
| 吸收面积、固定半径与空间块统计 | `case_analysis.py` | 02、03 |
| 系数可行性、配对RMSE差 | `diagnostics.py` | 03 |
| 七种失配、公开Landsat重新解混 | `validation_study.py` | 02、03 |
| 同日MODIS与原生雪标志评价 | `modis_validation.py` | 03 |
| 教学读取、中文表格、图例与制图 | `teaching.py` | 三份Notebook |
| 真实足迹相交、部分覆盖与MODIS QA | `geo.py`、`modis.py` | 完整输入准备 |
| 完整AVIRIS重提端元与三模型 | `scripts/rebuild_public_case.py` | L2 |
| HDF原生标识与字段提取 | `scripts/prepare_modis_hdf.py`、`matlab/export_modis.m` | L2 |
| 同日参考面积聚合与完整验证 | `scripts/prepare_modis_validation.py` | L2 |
| 小输入独立实验统一运行 | `scripts/run_validation_study.py` | 控制与Landsat |
| 暂存构建、HTML与练习副本 | `scripts/build_notebooks.py` | L1 |
| 数值产物、链接及真实浏览器 | `verify_artifacts.py`、`verify_site.py`、`browser_smoke.py` | 测试 |
| 报告图件 | `scripts/report_diagnostics.py`、`report_validation_figure.py`、`report_extended_figures.py` | 与Notebook共用输入 |
| MATLAB光学参考 | `matlab/`、`prepare_forward_reference.py`、`vendor/` | 01独立校核 |

未注明目录的算法模块位于`snowmix/`；验证脚本位于`scripts/`。历史物理、库谱和尺度记录与新增模型分开标注，来源与字段见`data/demo/README.md`。报告图号以PDF为准，输入和代码对应关系见`FIGURE_INDEX.md`。
