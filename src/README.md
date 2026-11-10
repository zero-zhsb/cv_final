# 项目源码与分工

- `data/`：A 主负责，共享名单读取、图像/真值与几何变换。
- `part1/`：A 主负责，模板差分、分数和校准；使用共享 reference。
- `part2/`：B 主负责，官方 AnomalyDINO 接入、名单适配、单图推理、预测导出。
- `evaluation/`：A 主负责，B 核对，共用指标、完整性检查和结果格式。

Part 2 适配器已实现并通过 CPU 小样例验证，见 [成员 B 工作记录](../docs/08_member_b_progress.md)。Part 1、data 公共加载器、evaluation 与正式 demo 仍待实现。详见 [A/B 操作步骤](../docs/02_team_workplan.md)。后续 Part 3 在选定实际问题后再建立源码模块。
