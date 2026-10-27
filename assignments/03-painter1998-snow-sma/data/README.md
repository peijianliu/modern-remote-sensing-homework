# 数据目录

`demo/` 是随仓库提供的教学输入，包含散射参数、独立太阳谱、参考子集、公开影像的代表谱和派生图层、配对统计及覆盖参考。WW80反照率从输入现算，完整MATLAB产物通过显式入口另行生成。

- 字段、单位、坐标系与处理说明：[demo/README.md](demo/README.md)
- 每个文件的来源和 SHA-256：[demo/manifest.json](demo/manifest.json)
- 完整输入与获取方式：[DATA_SOURCES.md](../DATA_SOURCES.md)
- 公开范围和外部许可：[PUBLIC_DATA_POLICY.md](../PUBLIC_DATA_POLICY.md)

`raw/` 与 `cache/` 用于完整流程和历史测试输入，由准备脚本生成并忽略提交。整景影像、登录信息、组员原始报告和再分发条件未确认的原谱不放入仓库。
