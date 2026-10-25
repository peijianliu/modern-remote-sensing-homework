"""依据已完成实验结果生成实验报告。"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from .artifacts import ROOT, configure_plots, write_json, paired_metrics
from .image import map_panel


def complete_runs(base, prefix):
    found = []
    for path in sorted(base.glob(prefix + "*")):
        if (path / "run_manifest.json").exists() and (path / "metrics.json").exists():
            manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest["status"] == "complete":
                found.append(
                    (
                        path,
                        json.loads((path / "metrics.json").read_text(encoding="utf-8")),
                        manifest,
                    )
                )
    return found


def link(path: Path, label: str):
    return f"[{label}]({path.as_posix()})"


def image(path: Path, alt: str):
    return f"![{alt}]({path.as_posix()})"


def build_report(config):
    configure_plots()
    base = ROOT / "experiments" / config["project"] / config["run_date"] / config["version"]
    physics = complete_runs(base, "01_physics")[-1]
    paper_runs = complete_runs(base, "02_03_paper")
    paper = [
        x
        for x in paper_runs
        if x[1]["sources_and_assumptions"]["solar_angle_assumption_deg"] == 0
        and x[1]["sources_and_assumptions"]["sensor"].startswith("2013")
    ][-1]
    angle60 = [
        x for x in paper_runs if x[1]["sources_and_assumptions"]["solar_angle_assumption_deg"] == 60
    ][-1]
    stability = complete_runs(base, "04_stability")[-1]
    images = complete_runs(base, "05_07_image")
    mammoth = [x for x in images if abs(x[1]["roi_center_lon_lat"][1] - 37.62848) < 0.003][-1]
    low_snow = [x for x in images if abs(x[1]["roi_center_lon_lat"][1] - 37.55625) < 0.003][-1]
    water_runs = [x for x in images if abs(x[1]["roi_center_lon_lat"][1] - 37.97292) < 0.003]
    water = water_runs[-1] if water_runs else None
    qa = complete_runs(base, "08_qa_sensitivity")[-1]
    controls = complete_runs(base, "09_resolution_control")
    controlled = controls[-1] if controls else None
    physical, p, s, q = physics[1], paper[1], stability[1], qa[1]
    verify = json.loads(
        (ROOT / "experiments/artifact_verification.json").read_text(encoding="utf-8")
    )
    report_root = ROOT / "reports"
    report_root.mkdir(exist_ok=True)
    number = 1
    while (report_root / f"实验报告_2026-10-01_v1_r{number:03d}.md").exists():
        number += 1
    report = report_root / f"实验报告_2026-10-01_v1_r{number:03d}.md"
    # 两个分离研究区按区域分层统计，水体区域单独排除。
    combined = []
    for path, metrics, _ in (mammoth, low_snow):
        table = pd.read_csv(path / "tables/coarse_pixel_comparison.csv")
        paired = table[table.paired_valid].copy()
        paired["roi"] = path.name
        combined.append(paired)
    combined = pd.concat(combined, ignore_index=True).drop_duplicates(["tile_row", "tile_column"])
    combined_path = report_root / f"双源配对像元_2026-10-01_v1_r{number:03d}.csv"
    combined.to_csv(combined_path, index=False)
    aggregate_metrics = paired_metrics(combined.aviris_mean_snow, combined.real_modis_snow)
    summary = {
        "strict_stratified_rois": [mammoth[0].name, low_snow[0].name],
        "strict_total_pixels": len(combined),
        "strict_reference_snow_ge_0p1_pixels": int((combined.aviris_mean_snow >= 0.1).sum()),
        "pooled_agreement_for_description_only": aggregate_metrics,
        "primary_snow_roi_below_n50": True,
        "water_roi_excluded_from_snow_validation": water[0].name if water else None,
    }
    write_json(report_root / f"结果摘要_2026-10-01_v1_r{number:03d}.json", summary)
    # 从已保存表格生成补充诊断图。
    for path, metrics, _ in (mammoth, low_snow):
        table = pd.read_csv(path / "tables/coarse_pixel_comparison.csv")
        name = "coarse_grain_comparison_um"
        if not (path / "figures" / f"{name}.png").exists():
            map_panel(
                path,
                name,
                [
                    table.aviris_snow_weighted_radius_um,
                    table.simulated_radius_um,
                    table.real_radius_um,
                ],
                ["AVIRIS雪面积加权参照", "模拟MODIS有效半径", "真实MODIS有效半径"],
                tuple(metrics["coarse_shape"]),
                (30, 1100),
                "viridis",
            )
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    for name, group in combined.groupby("roi"):
        label = "Mammoth有雪区" if name == mammoth[0].name else "南侧低雪区"
        ax.scatter(
            group.aviris_mean_snow,
            group.real_modis_snow,
            s=24,
            alpha=0.7,
            label=f"{label} n={len(group)}",
        )
    ax.plot([0, 1], [0, 1], color="black", lw=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="聚合AVIRIS模型雪覆盖",
        ylabel="真实MODIS模型雪覆盖",
        title="严格QA：有雪区与低雪区分层显示",
    )
    ax.legend()
    fig.savefig(report_root / f"双源分层散点_2026-10-01_v1_r{number:03d}.png", bbox_inches="tight")
    plt.close(fig)
    root_sources = [
        "1. [Painter et al. (1998), RSE 65(3), 320–332](https://doi.org/10.1016/S0034-4257(98)00041-8)。",
        "2. [Wiscombe & Warren (1980), snow albedo model I](https://www.atmos.washington.edu/~sgw/PAPERS/1980_WWI.pdf)。",
        "3. [SNICAR-ADv3 官方论文](https://gmd.copernicus.org/articles/14/7673/2021/)、[源码](https://github.com/mflanner/SNICARv3)。",
        "4. [USGS splib05a Quartz HS32](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/M/quartz_hs32.html)。",
        "5. [JPL AVIRIS L2 数据说明](https://aviris.jpl.nasa.gov/dataportal/20170911_AV_Download.readme)。",
        "6. [NASA Terra MODIS RSR](https://mcst.gsfc.nasa.gov/calibration/parameters)。",
        "7. [MOD09GA C6.1 场景](https://cmr.earthdata.nasa.gov/search/concepts/G2261932874-LPCLOUD.html)、[官方质量指南](https://www.earthdata.nasa.gov/s3fs-public/2025-04/MOD09_User_Guide_V61.pdf)。",
        "8. [MODIS Snow 官方算法说明，水体筛选](https://modis-snow-ice.gsfc.nasa.gov/?c=sug_main)。",
        "9. [Painter et al. (2003), MEMSCAG](https://doi.org/10.1016/S0034-4257(02)00187-6)。",
        "10. [Painter et al. (2009), MODSCAG](https://doi.org/10.1016/j.rse.2009.01.001)。",
    ]
    text = []
    text.extend(
        [
            "# Painter 1998 与课程作业复现实验报告",
            "",
            "记录日期：2026-10-01；实现：MATLAB R2024a + Python 3.11。",
            "",
            "## 1. 已完成内容与结论边界",
            "",
            "已运行纯雪物理正演、Fig. 3 已知真值模拟、多粒径端元选择、20次噪声重复、非负约束及端元遗漏对照，随后完成30 m AVIRIS、模拟500 m MODIS、真实同日MOD09GA解混及成图。误差项 E_c 与光谱 RMS 在所有模型中由原始系数重建。",
            "",
            "固定雪粒径导致覆盖偏差的趋势得到复现，历史输入差异列于方法表。真实双源对比已运行，但高质量有雪像元不足：Mammoth严格QA仅11个配对，其中3个AVIRIS模型雪比例≥0.1；74个南侧配对主要是低雪像元。101个保留邻云的像元只用于敏感性对照。这组实验评价双传感器模型一致性。",
            "",
            "## 2. 数据、模型与复现差异",
            "",
            "| 项目 | 本次实际采用 | 与原文/课件的关系 |",
            "|---|---|---|",
            "| 原始AVIRIS | 2013-05-03 f130503t01p00r14，224 bands，14.3 m；JPL L2 apparent surface reflectance | 非1994原场景；30 m是输出网格 |",
            "| 真实MODIS | 同日Terra MOD09GA.061 h08v05，7 bands | 独立观测；500 m产品实际网格间距约463.313 m |",
            "| 解混波段 | 合成224；实景182 | 现代掩膜，未恢复原文175波段索引 |",
            "| 雪正演 | SNICAR-ADv3球形、无杂质、100 m厚雪、200 kg/m³、Warren1984选项；另算2008选项 | 采用更新后的光学属性 |",
            "| Fig. 3正演 | WW80半无限delta-Eddington公式，光学属性来自SNICAR | 数学工作流复现；光学属性包含lognormal粒径分布 |",
            "| AVIRIS响应 | 实际2013中心波长/FWHM的Gaussian近似 | 1994完整RSR未取得 |",
            "| 非雪端元 | USGS HS32.4B quartz、Lodgepole pine针叶；ECOSTRESS Granite_H2、87P3855土壤 | 实验室材料代表谱：针叶、花岗岩和土壤 |",
            "| 砾石 | quartz_proxy仅作代理与可辨识性对照 | 未取得独立真实gravel谱，不报告已验证砾石覆盖 |",
            "",
            "AVIRIS原始header未写波长单位；依据365.9298–2496.236的数值与仪器范围解释为nm并转为μm，原header保留。MODIS实际属性已读取确认：scale=0.0001，reflectance fill=-28672，valid range=-100…16000，QC fill=787410671。HDF-EOS按YDim、XDim读取；未误用压缩附加观测的_c SDS。",
            "",
            "所有原始文件的来源和SHA-256保存在 "
            + link(ROOT / "data/sources", "来源清单")
            + "；两套大数据已经取得并保留。",
            "",
            "## 3. 算法与实验定义",
            "",
            r"\[R_c=\sum_i F_iR_{i,c}+E_c,\quad E_c=R_c-\widehat R_c,\quad RMS=\sqrt{M^{-1}\sum_cE_c^2}.\]",
            "",
            "原文模式用MGS求非阴影系数，允许负值与超过1；零阴影系数为1−ΣF_i。阴影归一化雪比例为 F_s/(1−F_shade)，原始系数用于光谱重建。FCLS另解非负、比例和为1的模型，用简单形活跃面枚举，并与独立SLSQP求解交叉核对。每个候选模型只放一个雪端元，再按RMS选择，逐候选求解。<sup>[1,9]</sup>",
            "",
            "合成真实半径50–1500 μm、步长10，真实雪比例0–1、步长0.01，共146×101×224。真值阴影系数为零，拟合含阴影补量。实景雪库30–1100 μm、步长10。归一化分母≤0.05及雪比例<0.1时不报告粒径；边界粒径另有QA标记。",
            "",
            "Snow spectral albedo为方向—半球量；实景传感器反射率按近似使用，方向性误差按观测几何差异讨论。太阳谱加权反照率积分0.2–5 μm。模拟MODIS采用一致的RSR卷积，空间聚合按四角投影多边形与MODIS像元相交面积加权，保留有效覆盖率。四角多边形用于局部投影边界近似，PSF影响列为后续实验。",
            "",
            "## 4. 物理图与原文模拟结果",
            "",
            f"60°下，100 μm代表新雪的加权反照率为{physical['albedo_60deg_100um']:.4f}，1000 μm代表老雪为{physical['albedo_60deg_1000um']:.4f}。新/老雪仅作半径代表标注。100与200 m雪层的最大反射率差为{physical['max_depth_100m_vs_200m_difference']:.1e}。曲线覆盖标准库30–1500 μm，未外推至课件5–2500 μm全轴。<sup>[2,3]</sup>",
            "",
            image(physics[0] / "figures/grain_radius_broadband_albedo.png", "雪粒半径与加权反照率"),
            "",
            image(physics[0] / "figures/snow_spectra_60deg.png", "60度雪光谱"),
            "",
            "| 条件 | 原文约值 | 本次0°假设 | 本次60°敏感性 |",
            "|---|---:|---:|---:|",
            f"| 50 μm纯雪，被200 μm端元解混 | 0.86 | {p['pure_snow_estimate_50um']:.4f} | {angle60[1]['pure_snow_estimate_50um']:.4f} |",
            f"| 1500 μm纯雪，被200 μm端元解混 | 1.20 | {p['pure_snow_estimate_1500um']:.4f} | {angle60[1]['pure_snow_estimate_1500um']:.4f} |",
            "",
            f"匹配200 μm时最大光谱RMS={p['matched_200um_max_rms']:.2e}；多粒径模型最大RMS={p['minimum_max_rms']:.2e}。固定模型雪比例RMSE={p['fixed_fraction']['rmse'] * 100:.3f}个百分点；多粒径在同模型、同库的无噪声真值上恢复比例及粒径。这项检验用于同库合成数据的数值实现验证。",
            "",
            "0°与60°改变端点数值，但本次仍未达到原文粗粒雪约1.2。光学常数/粒径分布、quartz样本与bandpass均有差异；目前未逐一隔离其贡献，不给某一个原因作确定归因。",
            "",
            image(paper[0] / "figures/fig3a_rms_heatmap.png", "粒径与比例的误差面"),
            "",
            image(paper[0] / "figures/fig3b_pure_snow_fraction.png", "固定雪端元覆盖偏差"),
            "",
            "## 5. 噪声、约束和端元遗漏",
            "",
            "使用75个控制案例（5种半径×5种雪比例×3种阴影），三个噪声标准差各20次，方法共享噪声以进行配对比较。区间为噪声重复的2.5%–97.5%分位，不当作实景精度置信区间。",
            "",
            "| σ | 方法 | 覆盖RMSE均值 / pp | 粒径RMSE / μm |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in s["summaries"]:
        text.append(
            f"| {row['noise_std']:.3f} | {row['method']} | {row['cover_rmse_mean'] * 100:.3f} | {row['radius_rmse_um_mean']:.2f} |"
        )
    text.extend(
        [
            "",
            f"遗漏松针端元时，真实雪比例0.5被估计为{s['omission']['omitted_snow']:.4f}；补回该端元后为{s['omission']['complete_snow']:.4f}。重复quartz端元被秩检查拒绝；近似重复端元受小噪声后各自比例明显波动，显示相似端元的比例估计对噪声敏感。",
            "",
            image(stability[0] / "figures/noise_and_constraints.png", "噪声与求解约束"),
            "",
            image(stability[0] / "figures/omitted_endmember_residual.png", "遗漏端元残差"),
            "",
            "## 6. 实景与尺度退化",
            "",
            "Mammoth AVIRIS固定200 μm、MIN RMS原文模式、MIN RMS FCLS的平均光谱RMS分别为"
            + ", ".join(
                f"{mammoth[1]['models'][key]['spectral_rms_mean'] * 100:.3f}%"
                for key in ["fixed_original", "multi_original", "multi_fcls"]
            )
            + "。原文模式在许多无雪像元产生负雪比例；FCLS提高部分像元光谱误差，同时给出物理可行比例。库端元不足、针叶/林冠差异及方向性差异列为待检验的误差来源。",
            "",
            image(mammoth[0] / "figures/aviris_snow_fraction.png", "三种AVIRIS雪覆盖结果"),
            "",
            image(mammoth[0] / "figures/aviris_grain_radius_um.png", "AVIRIS雪粒径图"),
            "",
            image(mammoth[0] / "figures/aviris_endmember_fractions.png", "植被岩石土壤端元比例"),
            "",
            "| ROI | 仅光谱退化RMSE / pp | 仅空间退化RMSE / pp | 光谱＋空间退化RMSE / pp | 同波段两种处理顺序RMSE / pp |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, m, _ in (mammoth, low_snow):
        text.append(
            "| "
            + m.get("roi_label", "Mammoth西侧")
            + " | "
            + " | ".join(
                f"{m[key]['rmse'] * 100:.3f}"
                for key in [
                    "spectral_degradation_agreement",
                    "spatial_only_agreement",
                    "combined_degradation_agreement",
                    "noncommutation_same_bands",
                ]
            )
            + " |"
        )
    text.extend(
        [
            "",
            "上表为早期应用分支：光谱退化时同时简化为雪/植被/裸地字典，分辨率与字典变化共同作用。后续使用固定字典分离两者影响，参照为AVIRIS模型估计。",
            "",
            image(mammoth[0] / "figures/controlled_degradation_snow.png", "尺度退化实验"),
            "",
            image(mammoth[0] / "figures/aviris_pixel_residuals.png", "实景像元误差项"),
            "",
            "## 7. 真实MODIS、QA敏感性与失败选区",
            "",
            "| 区域/策略 | 配对像元 | 覆盖差异RMSE / pp | R² = 1−SSE/SST | 解释 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for label, entry in [("Mammoth严格7波段", mammoth), ("南侧低雪区严格7波段", low_snow)]:
        m = entry[1]["real_modis_agreement"]
        text.append(
            f"| {label} | {m['n']} | {m['rmse'] * 100:.3f} | {m['r2']:.4f} | {'样本不足50，有雪样本更少' if m['n'] < 50 else '数量达标但雪比例范围不足'} |"
        )
    for tier in q["tiers"][1:]:
        m = tier["agreement"]
        text.append(
            f"| {tier['tier']} | {m['n']} | {m['rmse'] * 100:.3f} | {m['r2']:.4f} | {'保留邻云，只作敏感性对照' if tier['adjacent_cloud_retained'] else '固定5波段对照'} |"
        )
    text.extend(
        [
            "",
            f"严格两区合计{len(combined)}个不同配对像元，按区域分层保存；仅{summary['strict_reference_snow_ge_0p1_pixels']}个AVIRIS模型雪覆盖≥0.1。低雪区R²为负，表示误差大于用参照均值预测的基准；小绝对RMSE不说明积雪反演已经验证成功。",
            "",
            image(
                report_root / f"双源分层散点_2026-10-01_v1_r{number:03d}.png", "严格QA双源分层比对"
            ),
            "",
            image(qa[0] / "figures/qa_sensitivity_scatter.png", "QA策略敏感性"),
            "",
            "北侧NDSI候选区实际包含大面积水体。可见光图与低近红外反射率检查不支持“积雪丰富”的标签，该区域已退出积雪验证，原始失败运行保留。水体也可具有高NDSI，官方算法包含亮度/水体筛选；本轮后续选区检查采用green>0.15、NIR>0.11作为明确的额外候选筛选。<sup>[8]</sup>",
            "",
            "MOD09GA配对依据同日产品及观测指针，太阳/传感器角度一并保存。CMR的00:00是日产品时间；雪粒径图展示模型有效参数，绝对精度需同期实测。",
            "",
            "## 8. 输出、验证与后续条件",
            "",
            "每次运行包含配置、代码哈希、指标JSON、CSV、PNG/SVG及GeoTIFF。较后运行另保存源码zip；早期试运行保留原指标及图表。所有配对像元按瓦片行列定位。QA图标记无效输入、秩不足、小分母、低雪比例、粒径边界、条件数和比例越界。输出比例为模型光谱系数，quartz_proxy标识石英粉末材料代理。",
            "",
            f"数值检查覆盖匹配端元合成误差、填充值与缩放、非整数面积守恒；成果检查涉及 {sum(x['rasters_checked'] for x in verify['runs'])} 个 GeoTIFF 的 CRS、波段说明、FCLS 非负性与比例和、RMS 及 QA 类型。北侧候选区的水体在影像与近红外光谱检查中识别。",
            "",
            "仍需更合适的晴空有雪配对时相才能扩大严格积雪验证；需要独立地面粒径资料才能报告粒径绝对精度；1994原场景、原始响应/端元与照片缺失，限制逐点复现原文图6–9。本文粒径图采用库端元MIN RMS方法；原文Fig. 5采用Nolin吸收积分方法。<sup>[1,9,10]</sup>",
            "",
            link(ROOT / "README.md", "运行说明")
            + "；"
            + link(ROOT / "experiments/artifact_verification.json", "成果检查结果")
            + "；"
            + link(combined_path, "严格双源配对CSV")
            + "。",
            "",
            "## 参考资料",
            "",
        ]
    )
    if controlled:
        control = controlled[1]
        insertion = [
            "",
            "### 固定字典的分辨率对照",
            "",
            "所有分支均保持雪、植被、岩石、土壤及零阴影，相同粒径库与FCLS；仅改变波段或空间尺度。",
            "",
            "| 对照 | 雪比例差异RMSE / pp |",
            "|---|---:|",
        ]
        for title, key in [
            ("仅光谱退化", "spectral_only"),
            ("仅空间退化", "spatial_only"),
            ("光谱＋空间退化", "combined"),
            ("同MODIS波段、两种处理顺序", "same_bands_noncommutation"),
        ]:
            insertion.append(f"| {title} | {control[key]['rmse'] * 100:.3f} |")
        insertion.extend(
            [
                "",
                image(
                    controlled[0] / "figures/same_dictionary_resolution_control.png",
                    "固定字典尺度对照",
                ),
                "",
            ]
        )
        index = text.index("## 7. 真实MODIS、QA敏感性与失败选区")
        text[index:index] = insertion
    text.extend(root_sources)
    report.write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(
        ROOT / "reports/latest.json", {"report": str(report.relative_to(ROOT)), "summary": summary}
    )
    print("REPORT:", report.relative_to(ROOT), flush=True)
    return report
