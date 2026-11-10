# 项目源码与分工

- `data/`：A 主负责，共享名单读取、图像/真值与几何变换。
- `part1/`：A 主负责，模板差分、分数和校准；使用共享 reference。
- `part2/`：B 主负责，官方 AnomalyDINO 接入、名单适配、单图推理、预测导出。
- `evaluation/`：A 主负责，B 核对，共用指标、完整性检查和结果格式。

Part 2 适配器与完整矩阵由 B 完成，见 [B 的交接](../docs/10_part2_results.md)。Part 1 的 `TemplateInspector` 已实现，同样提供 `fit_references()` / `predict()`，返回 score、anomaly_map、inference_ms 等。公共评估入口是 `scripts/evaluate.py`，复用 B 的严格指标实现；公共地图格式加载器增加模板图分支，未改变 DINO 图重建语义。

A 当前只完成 cashew/1-shot/seed=0 的完整测试集与 CLI 演示，详见 [A 的首个里程碑](../docs/11_member_a_milestone.md)。Part 1 全矩阵、通用交互界面与 Part 3 不在本次完成范围。
