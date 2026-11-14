# CV Final · Few-Shot Industrial Visual Anomaly Detection

**期中视频素材已整理：** [素材包入口](presentation/midterm_video/README.md)，含 8 张 1080p 图卡、中英文讲稿、字幕草案、280 秒时间线和离线 HTML 翻页页。本次 Canvas 要求为不超过 5 分钟的阶段进展视频，无需先完成 Part 1 全量实验或联合实时界面。本次素材在 `codex/midterm-video-materials` 分支。

两人课程项目：先完成 **Part 1（模板差分）与 Part 2（AnomalyDINO）**，用于 Pre 展示；后续再完成 Part 3 的一个改进。代码与实验记录在 GitHub 协作，数据、权重和实验任务在服务器上运行。

## 当前状态

- [x] 仓库结构、课程要求、A/B 分工、服务器步骤与 Pre 验收文档。
- [x] 数据名单生成工具（校准集排除、官方参考图选择规则、掩码检查）。
- [x] 环境检查工具、输出协议与 GitHub Issue/PR 模板。
- [x] 使用服务器课程预置 VisA 完成官方划分与共享名单生成。
- [x] Part 1 模板差分实现、公共评估入口与单图演示；cashew/1-shot/seed=0 首个里程碑，见 A 的说明。
- [ ] Part 1 六类别、四 shots、三 seeds 全量实验。
- [x] Part 2 官方代码接入、名单/校准适配、CPU 小样例与单图官方一致性检查。
- [x] Part 2 全六类、四 shots、三 seeds 正式实验与 Part 2 精确评估（72/72）。
- [x] Part 2 结果表、曲线、五个真实失败案例、单图 CLI 演示和 B 的 Pre 讲稿。
- [ ] Part 1/Part 2 联合演示、两方法对比、正式幻灯片与录屏。
- [ ] Part 3 改进与最终报告。

**成员 B 的 Part 2 主实验已完成：11,520 条测试预测。** 8-shot 的六类图像 AUROC 为 **96.62±0.57%**，pixel AUROC 为 **98.21±0.02%**（三个 seed）。完整结果及局限见 [Part 2 结果与交接](docs/10_part2_results.md)。A 当前只完成单类别首个里程碑，不代表 Part 1 全量或 Part 3 已完成。

B 的交付位于 **`codex/part2-anomalydino`**；A 的首个里程碑在其基础上的 **`codex/part1-integration`** 分支。数据、权重、完整异常图保留在各自服务器工作目录。

## 从哪里开始

| 文档 | 用途 |
|---|---|
| [课程要求](docs/01_course_requirements.md) | 必做、可选、最终提交与 Pre 阶段的区别 |
| [A/B 详细分工](docs/02_team_workplan.md) | 每个人按顺序做什么、交付什么、如何验收 |
| [服务器操作手册](docs/03_server_runbook.md) | 环境、官方依赖、数据处理、后台任务与复现 |
| [实验与输出协议](docs/04_experiment_protocol.md) | 参考图选择、指标、阈值、文件格式与公平比较 |
| [Pre 展示清单](docs/05_pre_checklist.md) | 必备结果、建议讲解顺序、现场演示和备份 |
| [GitHub 协作](docs/06_github_workflow.md) | 分支、PR、任务清单与版本冻结 |
| [来源与适配记录](docs/07_sources_and_adaptations.md) | 课程依据、上游版本与必须记录的修改 |
| [成员 B 实际进展](docs/08_member_b_progress.md) | 真实数据路径、环境、可运行命令与验证边界 |
| [GPU 实验记录](docs/09_gpu_experiments.md) | 完整矩阵命令、compact 异常图、精确评价与效率说明 |
| [Part 2 结果与交接](docs/10_part2_results.md) | 已完成的成绩、错误分析、服务器复现和 A 的接入步骤 |
| [A 的首个里程碑](docs/11_member_a_milestone.md) | 独立 venv、cashew 单配置、公共评估、同图演示与复现 |
| [成员 B Pre 讲稿](presentation/member_b_notes.md) | 方法、实验、结果与答疑 |

## 第一个里程碑

**A：一个类别的 Part 1；B：同一类别的 Part 2。** 两人使用同一份参考图名单、同一批测试图和公共评估，输出能接入同一个演示。

然后扩展至六类别 × 四种 shot × 三个 seed：每种方法 72 个类别配置，两种方法共 144 个；这不是 144 次训练，方法均无需训练基础模型。

## 目录

```text
configs/             课程设置与方法配置
manifests/           校准图、参考图、测试图名单（可提交 JSON）
requirements/        本项目轻量依赖；模型环境另外配置
scripts/             名单、Part 2 推理/评价/汇总/演示工具
src/                 Part 2 实现；Part 1 和联合接口待 A 接入
demo/                已验证的 Part 2 单图样例；联合界面待完成
results/             可提交的指标表与精选可视化
docs/                详细操作文档
presentation/        Pre 讲稿与导出文件说明
report/              最终报告预留
tests/               数据隔离、Part 2 数值、异常图与评价检查
.github/             Issue 和 PR 模板
```

## 现在可以运行的工具

以下命令在本仓库根目录运行，Python 3.10+；名单工具仅使用标准库。

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -p test_prepare_manifests.py -v
python scripts/prepare_manifests.py --help
```

真实数据准备好后：

```bash
python scripts/prepare_manifests.py --data-root /srv/cv-final/data/VisA_pytorch/1cls --output manifests/visa_1cls.json
```

服务器实际路径由你们确定。不要把绝对路径写入共享配置；在运行时传入。模型命令、官方版本和课程适配步骤见服务器手册。

## 贡献

成员 A：数据、Part 1、公共评估、演示与对应文档。成员 B：模型环境、Part 2、实验汇总、效率测试与对应文档。双方共同核对协议、复现和讲解。真实姓名与工作记录由本人在 PR 中填写。

官方资源：[AnomalyDINO](https://github.com/dammsi/AnomalyDINO)、[VisA / spot-diff](https://github.com/amazon-science/spot-diff)、[DINOv2](https://github.com/facebookresearch/dinov2)。第三方代码需保留许可证与来源。

## 成员 B 实施入口

`scripts/run_part2.py` 读取共享名单，复用固定官方版本的特征提取、PCA masking、旋转与异常图渲染；独立校准，不把校准图加入 reference 或正式测试。默认 CPU、两个线程，以便共享服务器先做小样例功能检查。

正式结果和接入步骤见 [Part 2 结果与交接](docs/10_part2_results.md)，早期 CPU 检查见 [成员 B 工作记录](docs/08_member_b_progress.md)。共享 GPU 上本次实验已运行结束。Part 1 与公共演示由 A 接入；Part 2 原始浮点输出和已完成评估可供交叉核对。
