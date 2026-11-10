# 真实指标表

只放真实实验导出的 CSV：逐配置指标、每类别/shot 的 mean±std、六类平均、时间和显存。每行关联 run_id、manifest_id、commit。不放占位成绩。

本次完整主结果是 `part2_gpu_*`：72 配置、11,520 条测试预测、24 组类别/shot 汇总与 4 行整体汇总。逐配置表和 run JSON 记录运行来源；聚合表与同一目录的 run 元数据配套使用。AUROC CSV 数值范围为 0–1，展示为百分比时乘 100，标准差为 ddof=0。

`part2_cpu_smoke_*`、`part2_gpu_preflight.json` 是早期样例或单类别预检，不能替代主结果。`cpu_gpu_configuration_parity.csv` 为两个后端的 19 个完整重叠配置核验；其他 parity/validation 文件记录数值和指标验证。

详细解释、图表、失败分析和服务器复现步骤见 [Part 2 结果与交接](../../docs/10_part2_results.md)。
