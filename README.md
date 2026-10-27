# 国科大现代遥感技术与应用作业

可运行的遥感课程案例，提供方法说明、代码、公开教学数据与计算结果。

## Painter 1998 雪粒径与光谱混合分析

研究雪粒径变化如何影响反射率、光谱混合解混和像元雪覆盖估计。案例以经典论文的流程为主线，结合五份小组作业的实验设计，在统一的 2013 年 AVIRIS / MODIS 数据上补充对照，并以 2016/2017 年同日公开覆盖参考扩展独立验证。

- [作业说明与运行步骤](assignments/03-painter1998-snow-sma/README.md)
- [HTML 案例入口](https://peijianliu.github.io/modern-remote-sensing-homework/)
- [Notebook 源文件](assignments/03-painter1998-snow-sma/notebooks/)
- [综合报告](assignments/03-painter1998-snow-sma/reports/)
- [公开数据与许可](assignments/03-painter1998-snow-sma/PUBLIC_DATA_POLICY.md)
- [成员贡献](CONTRIBUTORS.md)

默认教学流程使用仓库内的小型数据，无需 MATLAB、Earthdata 登录或完整航带下载。完整遥感处理的输入、依赖及 MATLAB 参考入口见[数据来源与重建约定](assignments/03-painter1998-snow-sma/DATA_SOURCES.md)。

原创代码采用 MIT；报告和教学说明采用 CC BY 4.0。第三方模型和外部数据保留各自许可。

## 学习路线

- 30 分钟：读 HTML 01 的单波长手算与雪谱，再看综合报告的 Fig. 3 控制实验。
- 3 小时：克隆仓库，运行三份 Notebook，并在独立工作副本中改半径、角度与面积阈值。

Git 中保留无输出 Notebook 和唯一静态源 `site/`；执行版、HTML 及运行记录由 Actions 构建为 artifact。默认构建位于忽略的 `build/site/`，不会回写源码。引用本案例可使用 `CITATION.cff`。

实验结果与各分支的适用范围见[实验说明](assignments/03-painter1998-snow-sma/EXPERIMENTS.md)。MGS有符号系数、FCLS可行比例、光谱残差和独立覆盖误差分别展示。
