# 来源与课程适配记录

## 课程来源

用户提供的 P2 PDF：`Project_2__Few_Shot_Industrial_Visual_Anomaly_Detection.pdf`。

- 第 1–2 页：任务概述、两人分组、报告/代码/demo 提交要求、公平比较。
- 第 3 页：Part 1 模板差分和校准、Part 2 AnomalyDINO 设置。
- 第 4 页：Part 3 可选改进方向、六类 VisA、指标和不允许测试泄漏的规则。
- 第 5 页：参考文献。

该 PDF 作为课程要求依据。课程文档未给出 Pre 独立 rubric、具体百分比和最高分数值门槛。

## 外部资源（2026-11-09 核对）

| 资源 | 链接 | 记录 |
|---|---|---|
| AnomalyDINO | https://github.com/dammsi/AnomalyDINO | 检查时 HEAD `b9d1c2648e3a5247437d4d953d907a8f3d994457`；官方复现代码，Apache-2.0 |
| VisA / spot-diff | https://github.com/amazon-science/spot-diff | 检查时 HEAD `2a692ab575001cbde74d402d897a7286086c6199`；下载、1cls、掩码 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | 预训练 backbone 与权重来源；实际运行记录 torch hub 所用版本 |
| PyTorch 安装 | https://pytorch.org/get-started/locally/ | 按实际服务器驱动选择，不声称已验证你们的 CUDA 环境 |

官方 AnomalyDINO `run_anomalydino.py` 支持 `--resolution`、`--shots`、`--num_seeds`、`--preprocess`、`--eval_segm` 和 `--faiss_on_cpu`。`src/detection.py` 的 seed 选择是排序切片；`src/backbones.py` 使用短边缩放和 patch 倍数裁剪。本仓库文档据此给出适配说明。

## 课程适配状态

- [x] Part 2 限制为共享 manifest 允许的六类。
- [x] 在 reference 选择前排除 calibration。
- [x] Part 2 读取共享 JSON，导出实际名单；Part 1 待 A 接入。
- [x] 独立处理校准图，不混入记忆库和正式测试行。
- [x] 导出公共格式的原始 score/map。
- [ ] 完成公共评估与上游结果核对、完整时间/显存测量。
- [x] 提交自己编写的适配源码和版本信息，不修改上游 detector。

适配器复用上游 backbone 的 prepare_image / extract_features / PCA masking，以及 utils 的 rotation / dists2map。仅覆盖模型加载来源，使用本地固定 DINOv2 源码与权重；近邻循环与导出由本项目实现。单张真实图与官方 detector 的对比见 [parity 记录](../results/tables/part2_parity.json)，不可据此宣称所有配置完全复现或达到论文成绩。

新增第三方代码、图、权重、数据或组件时，补充来源、许可证、修改点与论文引用。当前仓库没有拷入第三方方法代码，也没有假定可以为其重新授权。
