"""根据已完成实验的统计结果生成补充实验报告。"""

from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowmix.artifacts import ROOT, write_json
from scripts.verify_supplement import latest_complete


def read(path: Path) -> dict:
    """读取已保存的 JSON 结果或元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def link(path: Path, title: str) -> str:
    return f"[{title}]({path.as_posix()})"


def fig(path: Path, title: str, caption: str) -> str:
    return f"![{title}]({path.as_posix()})\n\n**{title}。** {caption}\n"


def run() -> Path:
    paths = {
        stage: latest_complete(stage)
        for stage in [
            "10_publication_figures",
            "11_empirical_spectra",
            "12_aviris_band_sensitivity",
            "13_worldview_validation",
            "14_empirical_mixtures",
            "15_review_reanalysis",
            "16_conservative_maps",
        ]
    }
    metrics = {
        stage: read(path / "metrics.json")
        for stage, path in paths.items()
        if (path / "metrics.json").exists()
    }
    verification_path = (
        ROOT / read(ROOT / "experiments/supplement_verification_latest.json")["verification"]
    )
    verification = read(verification_path)
    if verification["errors"]:
        raise ValueError("Artifact verification is not clean")
    for stage, path in paths.items():
        if (ROOT / verification["selected_runs"][stage]).resolve() != path.resolve():
            raise ValueError(
                "Verification is stale for selected outputs; rerun verify_supplement.py"
            )
    number = 1
    while (ROOT / f"reports/补充实验报告_2026-10-02_v2_r{number:03d}.md").exists():
        number += 1
    output = ROOT / f"reports/补充实验报告_2026-10-02_v2_r{number:03d}.md"
    lab = metrics["11_empirical_spectra"]
    band = metrics["12_aviris_band_sensitivity"]
    world = metrics["13_worldview_validation"]
    mixture = metrics["14_empirical_mixtures"]
    reanalysis = metrics["15_review_reanalysis"]
    catalog = read(paths["10_publication_figures"] / "figure_catalog.json")
    text = [
        "# Painter 1998：补充实验报告",
        "",
        "报告日期：2026-10-02。v2 批次按实验开始日期 2026-10-01 归档；运行起止时间、配置及源码哈希保存在 run_manifest.json。",
        "",
        "## 1. 方法与完成内容",
        "",
        "实验采用 MGS 解混、零光谱阴影补量和逐粒径候选的最小光谱 RMS 选择。误差项 E_c 由原始系数重建。数值检查覆盖已知比例恢复、近共线求解、配置执行和有效面积筛选。",
        "",
        "术语：MGS为modified Gram–Schmidt最小二乘；FCLS为fully constrained least squares，要求非负且各比例和为1；RMS为光谱残差均方根；fSCA为像元雪覆盖面积比例；SSA为雪的冰—空气界面面积/冰质量（比表面积）。覆盖误差百分点与反射率百分点分别标注，粒径统一指半径。",
        "",
        "增加粒径候选改善了部分光谱拟合，外部覆盖精度的变化按独立参考结果评价。公开实测雪谱和WorldView参考使原先仅靠自生成真值与双模型一致性的验证，扩展为真实观测谱挑战、控制混合和独立覆盖参考比较。原2013 AVIRIS/MODIS严格有雪样本不足的边界仍保留，新增Landsat实验单独评价。",
        "",
        "图件入口："
        + link(paths["10_publication_figures"] / "图集与图解.md", "完整注释图集")
        + "。",
        "",
        "## 2. 数值与制图检查",
        "",
        "| 问题 | 已完成修正与验证 | 对结论的影响 |",
        "|---|---|---|",
        "| 原文模式负雪比例与零雪同色；粒径边界未突出 | 新图使用<0/>1专色、NoData图例与分类QA；FCLS与MGS分开 | 旧Mammoth约58.6%负雪比例；原文模式粒径非空结果68.5%在库边界、95.4%存在某个比例越界，需结合越界与边界QA解释 |",
        "| 尺度对照只检查输入谱有效面积 | 同时要求归一化参照≥95%有效面积；粒径另要求≥95%已估计雪面积有粒径 | 两区各剔除1个参照支持不足粗像元；原严格真实MODIS配对未改变 |",
        "| JSON半径范围、固定半径、比例步长或冰选项被忽略 | 正式读取字段；未缓存半径明确报错；缓存雪形、密度、雪深、底层与太阳谱选项不符时拒绝 | 默认旧运行参数一致，不追溯推翻；避免后续manifest与执行不一致 |",
        "| 近共线FCLS使用正常方程和范数差造成数值消减 | 病态子字典采用MGS算子构造边界方向及直接残差目标；保留条件数标记 | cond≈4.29×10⁷的无噪声探针从[0,0.6]恢复[0.2,0.4]；该探针用于近共线条件下的数值检验 |",
        "| 图件缺少坐标/单位/图例/解释，部分标签遮挡 | 14主图重绘；补实验图统一300 dpi PNG/SVG；分类边界图例及SSA编号 | 地图按原始网格nearest显示 |",
        "",
        "原文Fig.3粗粒端正文给出“nearly 1.2”，读图约1.16，均是近似参照。现代输入下纯雪50/1500 μm估计为0.8788/1.0957；1994响应、原光学属性及完全相同quartz输入未恢复。本文报告流程与趋势复现及历史输入造成的数值差异。<sup>[1]</sup>",
        "",
        "## 3. 公开数据",
        "",
        "| 数据 | 本轮实际使用 | 可以检验什么 | 配对与限制 |",
        "|---|---|---|---|",
        "| 作者实验室雪谱，Zenodo 3580078 | 10雪况均值、158重复样本及7组IceCube SSA；主分析用均值 | 真实谱与模型失配；条件级粒径参照 | 灯天顶55°/nadir；DHR与BRF不同；SSA最高记录点仅近表层代理，4组干雪 |",
        "| 作者野外谱，Zenodo 3580825 | 87目标、350–2500nm；82雪/阴影雪、5非雪 | 雪存在与外部背景挑战 | 日期/地点共17组；人工毫米GrainSize和周围SnowPatch不作为视场光学半径/雪比例真值 |",
        "| 作者WorldView fSCA，Zenodo 4031446 | 已下载9张30m参考及2张463m参考；本轮用2场 | 独立可见积雪覆盖评价 | 2016-02-13同日；2017-05-21对应Landsat05-22，隔1天；按这两场独立迁移实验评价 |",
        "| 匹配Landsat C2 L2与官方OLI响应 | 两场18个窗口文件、B2–7及QA；官方RSR | 将固定方法迁移到独立覆盖参考 | 按实际CRS/格网匹配；DN×0.0000275−0.2；保留雪位，负值和>1反射率不clip |",
        "",
        "公开资料的来源、许可、版本、字段和配对要求见 "
        + link(ROOT / "DATA_SOURCES.md", "数据来源说明")
        + "。WorldView 参考定义为可见雪覆盖，评价考虑分类与配准不确定性。<sup>[2–5]</sup>",
        "",
        "## 4. 独立实测谱与SSA的结果",
        "",
        "| 方法 | 6种类型一致干雪全雪目标：覆盖RMSE / pp | 4组SSA代理：粒径差异RMSE / μm | 所有10雪况平均光谱RMS / pp |",
        "|---|---:|---:|---:|",
    ]
    for label, item in lab["lab_methods"].items():
        text.append(
            f"| {label} | {item['clean_dry_cover_vs_known_full_target']['rmse'] * 100:.3f} | {item['dry_condition_SSA_radius_discrepancy']['rmse']:.2f} | {item['spectral_rms_mean'] * 100:.3f} |"
        )
    text += [
        "",
        "M_rimed14均值与逐样本元数据的干/湿标记冲突，已排除干雪主统计但保留光谱。158个重复样本按所属雪况分组。多粒径MGS在真实纯雪目标上仍可能分配负背景；FCLS改善这些目标的覆盖估计，但SSA参照只有4种干雪条件，作为条件级粒径参照。SSA按 r=3/(917·SSA) 换算表面积/体积等效半径。<sup>[2,6]</sup>",
        "",
        fig(
            paths["11_empirical_spectra"] / "figures/independent_lab_cover_and_ssa.png",
            "图1 独立雪样覆盖与SSA代理",
            "左图是真实全雪目标的固定/多粒径估计，附落叶条件另列；右图为近红外谱形半径与条件级SSA代理的对照，使用log轴辨读干雪聚集点，数字编号名称见该运行README。7 组中 4 组干雪、3 组湿雪；两类测量的方向性与采样深度存在差异。",
        ),
        fig(
            paths["11_empirical_spectra"] / "figures/independent_lab_observed_and_fitted.png",
            "图2 真实观测与拟合谱",
            "四种雪况包括新雪、干雪、湿雪和落叶湿雪；缺测窗断开。分别比较固定200μm与多粒径的光谱残差和覆盖误差。",
        ),
        "野外87个目标在预定0.5雪存在阈值下为82/82雪与5/5非雪一致；非雪仅5个且相邻样本相关，结果用于本数据集的目标存在诊断。野外角度采用60°假设；人工粒径保留原测量定义。",
        "",
        "## 5. 实测端元的控制混合：检验理想结果能否迁移",
        "",
        "使用4种有SSA代理的干雪谱与5条独立非雪谱，生成440个已知比例控制案例（11个雪比例×2种灰阴影）。拟合使用冻结雪/松针/花岗岩/土壤库，真值背景来自独立实测谱。三个噪声水平0/0.005/0.01，各报告20重复、共享噪声；零噪声实际只拟合一次。每个雪况/背景组合等权，非零噪声区间只描述固定输入扰动。",
        "",
        "| 零噪声方法 | 正雪覆盖bias / pp | RMSE / pp | 零雪误检 / 40例 |",
        "|---|---:|---:|---:|",
    ]
    for item in mixture["noise_intervals"]:
        if item["noise_std"] == 0:
            text.append(
                f"| {item['method']} | {item['cover_bias_mean'] * 100:.3f} | {item['cover_rmse_mean'] * 100:.3f} | {item['false_positives_mean']:.0f} |"
            )
    text += [
        "",
        "这组已知系数数据不由反演雪库生成，因而能暴露光谱模型/背景失配。其系数误差评价跨测量几何、假设线性混合下的模型失配。",
        "",
        fig(
            paths["14_empirical_mixtures"] / "figures/empirical_mixture_cover_curves.png",
            "图3 外部光谱的覆盖偏差",
            "横轴为设定面积比例，纵轴为阴影归一化估计。实线中位数、浅色带四分位范围；越界值不clip。固定与多粒径模型都可系统偏离1∶1，FCLS减少但未消除失配误差。",
        ),
        "## 6. 独立WorldView覆盖参考：分场景、分层和共同样本",
        "",
        "每场在共同影像QA和参考有效面积≥0.95的区域，用固定seed抽12000像元；未按低误差或高雪挑点。雪/松针/花岗岩/零阴影字典、30–1100μm库和FCLS固定。六波段与四波段(B3/B5/B6/B7)均预定，四方法共同归一化有效子集用于公平对比，失效样本单独报告。空间区块50×50像元（1.5km），bootstrap 1000次。",
        "",
        "| 参考与观测日期 | 共同有效/预定 | 固定200六波段 RMSE / pp | 多粒径六波段 RMSE / pp | 多−固定 ΔRMSE及区块95%区间 / pp |",
        "|---|---:|---:|---:|---|",
    ]
    for scene in world["scenes"]:
        fixed = next(
            m
            for m in scene["methods"]
            if m["method"] == "fixed200_six_bands" and m["stratum"] == "all"
        )["common_methods_cover"]
        multi = next(
            m
            for m in scene["methods"]
            if m["method"] == "multi_six_bands" and m["stratum"] == "all"
        )["common_methods_cover"]
        delta = scene["paired_multi_minus_fixed"]["six_bands"]
        interval = delta["rmse_delta_interval_95"]
        text.append(
            f"| {scene['reference_date']} → {scene['landsat_date']} (+{scene['day_offset']}d) | {multi['n']}/12000 | {fixed['rmse'] * 100:.3f} | {multi['rmse'] * 100:.3f} | {delta['multi_minus_fixed_rmse'] * 100:+.3f} [{interval[0] * 100:+.3f}, {interval[1] * 100:+.3f}] |"
        )
    text += [
        "",
        "两个六波段实验中，多粒径的总体覆盖误差略增，区块区间支持这一小差别。四波段多/固定差值区间包含0，未观察到稳定优势。区间描述现有场景与参考下的空间抽样差异；参考分类、BRDF、地形和时差另作误差来源讨论。",
        "",
        "| 场景 | 多粒径六波段：无雪n | 混合n与RMSE / pp | 近纯雪n与RMSE / pp |",
        "|---|---:|---:|---:|",
    ]
    for scene in world["scenes"]:
        strata = {
            m["stratum"]: m["common_methods_cover"]
            for m in scene["methods"]
            if m["method"] == "multi_six_bands"
        }
        text.append(
            f"| {scene['landsat_date']} | {strata['no_snow']['n']} | {strata['mixed']['n']} / {strata['mixed']['rmse'] * 100:.3f} | {strata['full_snow']['n']} / {strata['full_snow']['rmse'] * 100:.3f} |"
        )
    text += [
        "",
        "2017的总体RMSE约5.5pp，主要受无雪样本占多数影响；混合雪层仍约23pp误差。2016包含大量雪/混合雪，独立覆盖检验远比旧2013双模型参照充分，同时揭示了迁移欠缺。2016 四方法共同有效样本为 10763/12000，2017 为 11910/12000；其余为归一化失效样本，单独计数。",
        "",
        fig(
            paths["13_worldview_validation"]
            / "figures/LC08_L2SP_043033_20160213_02_T1_stratified_scatter.png",
            "图4 2016同日独立覆盖分层散点",
            "独立参考无雪<0.1、混合0.1–0.9、近纯雪≥0.9。四个方法展示所有各自归一化有效的预定样本；上表统计额外限制四方法共同子集。黑线1∶1；预定有效样本的偏差完整显示。",
        ),
        "WorldView提供可见雪覆盖参考，两场Landsat用于第三传感器迁移检验。后续空间粒径评价拟采用SnowEx光谱/SSA资料，需核对日期、足迹及测量层位。",
        "",
        "## 7. 波段、冰常数和入射角的诊断",
        "",
        "从解混前有效AVIRIS像元随机抽1200个、61空间块。保守窗口固定为0.42–1.30、1.50–1.75、2.05–2.35μm（150波段）；每5个保守波段留出1个，共30波段，所有模型均不拟合这些相同波段。与训练RMS比不同，公共留出比较能检验对未参与拟合光谱的预测。",
        "",
        "| 方法 | 公共留出RMS均值 / 反射率pp | 粒径库边界率 / 全1200样本 |",
        "|---|---:|---:|",
    ]
    for label, item in band["methods"].items():
        text.append(
            f"| {label} | {item['heldout_rms_mean'] * 100:.3f} | {item['radius_boundary_rate'] * 100:.2f}% |"
        )
    narrow = band["methods"]["conservative_train"]
    interval = narrow["paired_holdout_rms_delta_block_bootstrap"]["interval_95"]
    text += [
        "",
        f"保守窗口公共留出RMS由1.444降到1.272pp，空间块平均差95%区间[{interval[0] * 100:.3f}, {interval[1] * 100:.3f}]pp；覆盖与宽模型差异RMSE为{narrow['cover_difference_vs_wide']['rmse'] * 100:.3f}pp。公共留出结果支持光谱预测改善；覆盖和粒径按各自参照评价。角度60°导致库边界明显堆积，提示未校正局地地形入射角时粒径解释有风险；0°用于预定的角度敏感性对照。",
        "",
        fig(
            paths["12_aviris_band_sensitivity"] / "figures/withheld_spectral_band_sensitivity.png",
            "图5 预定波段与公共留出检验",
            "左图逐波段误差用于观察水汽窗邻近异常，缺测区域断线；灰区是预定保守排除窗口。右图所有方法都用相同30个未拟合波段评分，误差降低与方法间覆盖差别分别记录。",
        ),
        "另已把预定150波段用于全ROI FCLS，共62908有效输入，输出各类比例、阴影、候选粒径、RMS和QA GeoTIFF。旧与新地图共同区域覆盖差异RMSE约2.599pp；本轮增加native面积筛选，边缘有效性也略有变化。两幅地图用于波段与边缘筛选敏感性比较。",
        "",
        fig(
            paths["16_conservative_maps"] / "figures/conservative_grain_rms_boundary.png",
            "图6 全ROI保守波段诊断",
            "粒径在低雪或小分母时缺失，灰色明确标记；模型评分采用公共留出波段。右侧为分类边界标记，粒径有效性同时检查雪比例与归一化分母。图件按原30m像元显示。",
        ),
        "## 8. 面积支持修正、原文差异与图解交付",
        "",
        "| v1来源 | 仅空间旧→修正 RMSE / pp | 组合旧→修正 RMSE / pp | 粒径参考旧→支持充分n |",
        "|---|---:|---:|---:|",
    ]
    for label in ["05_07_image_r001", "05_07_image_r002"]:
        item = reanalysis[label]
        text.append(
            f"| {label} | {item['spatial_only_before']['rmse'] * 100:.4f} → {item['spatial_only_corrected']['rmse'] * 100:.4f} | {item['combined_before']['rmse'] * 100:.4f} → {item['combined_corrected']['rmse'] * 100:.4f} | {item['old_grain_reference_n']} → {item['corrected_grain_reference_n']} |"
        )
    text += [
        "",
        "粗像元粒径的支持条件变严后，Mammoth有效参照由183降至87，低雪区由92降至36；其余保留为空值，未插值。参照定义为高分模型的雪面积加权半径；粗像元反演参数为光学等效半径。原文/课件的粒径图可以在候选、地物字典、入射角及校正不同情况下有显著差别；图件按原数值及统一色标比较。",
        "",
        "主图集包含14张图，PNG 300 dpi、SVG，补齐坐标、投影、真北、比例尺、单位和NoData，MGS负值/超限与粒径库边界有专用显示。每张图下给内容、来源、读数和适用限制；以下为粒径与质量图示：",
        "",
        fig(
            ROOT / catalog[7]["png"],
            "图7 历史粒径与质量标记并列",
            catalog[7]["caption"] + " " + catalog[7]["limits"],
        ),
        "完整图集："
        + link(paths["10_publication_figures"] / "图集与图解.md", "图集与逐图解释")
        + "。补实验每组README也有每图说明，目录清单与源数值见附录。",
        "",
        "## 9. 验证结果、失败记录与下一轮优先级",
        "",
        f"检查范围包括近共线求解、配置执行、SSA 单位和空间区块统计。成果检查涉及 {verification['source_files_checked']} 份公开源文件 SHA、{verification['source_snapshots_checked']} 份源码快照、{verification['rasters_checked']} 张 GeoTIFF 和 {verification['pngs_checked']} 张 300 dpi PNG/SVG；WorldView 分层指标与 CSV、比例约束及 E_c/RMS 按保存结果核对。检查结果："
        + link(verification_path, "机器可读检查清单")
        + "。",
        "",
        "报告采用 run_manifest.json 中状态为 complete、且与成果检查版本一致的实验结果。配置、源码快照和运行记录随实验版本保存。",
        "",
        "课程要求的物理图、线性解混、误差项、各端元/阴影/粒径/QA成图、受控尺度变化和真实双源一致性均已有实际成果；本轮又补外部实测谱与独立覆盖参考。结果支持流程与趋势复现，外部观测补充了方法适用范围的证据。",
        "",
        "| 后续优先项 | 推荐实验与数据 | 为什么值得做 |",
        "|---|---|---|",
        "| 1 场景背景与阴影字典 | 现有两场Landsat以及余下7张WorldView参考；独立空间区块提取本地背景，与冻结光谱库在另一场/区块评价 | 外部实测混合已证明非雪谱会被补偿成雪；提取区与评价区必须分离，避免标签调库 |",
        "| 2 方向性/地形与辐射异常 | 同景DEM、真实局地入射角；保持参考不变，单独比较BRDF/地形处理；按现有aerosol QA与>1反射率分组 | 2016混合及近纯雪仍明显低估；各机制需单独对照实验量化 |",
        "| 3 多场景配准与尺度预算 | WorldView多场景，预定±30/60m位移和PSF敏感性；独立463m参考只与实际同日观测配对 | 像元支持、亚像元配准与PSF需分别检查 |",
        "| 4 光学粒径的独立成套测量 | SnowEx AVIRIS-NG与SSA场测，严格核对日期、足迹、测量深度和半径/直径 | 当前SSA含4个干雪条件的仪器参照；空间粒径评价需扩大同期采样 |",
        "| 5 原文历史输入与真实砾石 | 1994原AVIRIS响应/光学常数与现场背景资料；有颗粒尺度与测量条件的实际砾石谱 | 记录历史与现代输入差异；quartz按粉末材料代理解释 |",
        "",
        "上述后续实验用于量化背景、方向性、配准及粒径参考对结果的影响。",
        "",
        "## 附录：重跑与成果入口",
        "",
        "```powershell\n& .\\.venv\\Scripts\\python.exe -m snowmix supplement\n& .\\.venv\\Scripts\\python.exe -m snowmix review-report\n```",
        "",
        "缓存已准备时supplement仅新建运行目录；缺少新公开数据或特定角度缓存时会下载公开资料/运行MATLAB。完整冷启动还需要v1数据准备，AVIRIS大文件下载不含在30分钟缓存复现目标内。review-report只校验已完成结果并生成新版本报告。",
        "",
        "| 阶段 | 本次选取的完成版本 | 解释与数据入口 |",
        "|---|---|---|",
    ]
    for stage, path in paths.items():
        entry = path / ("图集与图解.md" if stage == "10_publication_figures" else "README.md")
        text.append(f"| {stage} | {path.name} | {link(entry, '图下注、配置及源数据说明')} |")
    text += [
        "",
        "## 参考来源",
        "",
        "1. [Painter et al. (1998)](https://doi.org/10.1016/S0034-4257(98)00041-8)。",
        "2. [Hannula & Heinilä：实验室雪谱/SSA](https://doi.org/10.5281/zenodo.3580078)。",
        "3. [Heinilä et al.：野外雪/裸地光谱](https://doi.org/10.5281/zenodo.3580825)。",
        "4. [Stillinger et al.：WorldView可见积雪覆盖验证参考](https://doi.org/10.5281/zenodo.4031446)。",
        "5. [USGS Landsat C2 L2缩放](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products)、[QA](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)、[官方OLI RSR](https://landsat.usgs.gov/spectral-characteristics-viewer)。",
        "6. [Warren：SSA与等效半径定义](https://doi.org/10.1098/rsta.2018.0161)。",
        "7. [官方SNICAR](https://github.com/mflanner/SNICARv3)、[SPIRES两流公式](https://github.com/edwardbair/SPIRES/blob/bcca68ebad36a34bf67f8bc88b7198eda9fc608f/SolarLongwaveRadiation/twostream.m)。",
    ]
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    summary = {
        "report": str(output.relative_to(ROOT)),
        "selected_runs": {stage: str(path.relative_to(ROOT)) for stage, path in paths.items()},
        "verification": str(verification_path.relative_to(ROOT)),
        "external_cover_reference_acquired": True,
    }
    write_json(output.with_suffix(".json"), summary)
    old = ROOT / "reports/latest.json"
    if old.exists() and not (ROOT / "reports/latest_v1.json").exists():
        (ROOT / "reports/latest_v1.json").write_bytes(old.read_bytes())
    write_json(old, summary)
    write_json(
        ROOT / "experiments/checkpoint.json",
        {
            "status": "supplement_complete",
            "report": str(output.relative_to(ROOT)),
            "selected_runs": summary["selected_runs"],
            "next_priorities": [
                "spatially held-out local background endmembers",
                "terrain/BRDF and radiometric sensitivity",
                "matched spatial grain-size measurements",
            ],
        },
    )
    print("REPORT", output)
    return output


if __name__ == "__main__":
    run()
