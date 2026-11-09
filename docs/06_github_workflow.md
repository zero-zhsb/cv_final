# GitHub 协作方式

仓库：https://github.com/zero-zhsb/cv_final

## 分支与目录责任

- `main`：两人已检查、能按当前说明复现的版本。
- A：`codex/part1-baseline`，主改 `src/part1/`、`src/data/`、`src/evaluation/`、`demo/`。
- B：`codex/part2-anomalydino`，主改 `src/part2/`、`patches/`、模型环境与结果汇总。
- 两人改公共配置、名单或输出 schema 前，先在对应 Issue 商定；合并后另一人同步。
- 初始骨架可能由助手提交；A/B 后续使用自己的身份提交实际贡献，不把助手初始化计作个人实验成果。

首次 clone 后从 main 创建各自分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/part1-baseline
```

B 替换分支名。完成一个可以验证的功能后，先 `git status`，只 stage 相关源文件和文档，不提交服务器数据/凭据。

```bash
git add src/part1/ docs/
git commit -m "Implement template-difference baseline"
git push -u origin codex/part1-baseline
```

然后在 GitHub 建 PR 到 main，填写运行命令、样例、验证结果和剩余问题。由另一人检查后合并；新任务可从更新的 main 开新分支。发生冲突时看懂两边改动再解决，不用强推覆盖对方成果。

## 建议先创建的 Issues

| ID（团队编号） | 标题 | 负责人 | 验收 |
|---|---|---|---|
| DATA-01 | 官方数据准备与共享名单 | A；B 审核 | 六类、72 套、校准隔离、完整 masks |
| P1-01 | 单类别模板差分与校准 | A | 指定同图输出与可解释流程 |
| EVAL-01 | 公共 AUROC 与聚合 | A；B 核对 | 正常图参与、跨 seed/类别统计正确 |
| ENV-01 | 服务器模型环境 | B | CUDA 推理、固定上游、依赖清单 |
| P2-01 | 官方代码课程适配 | B | 共享名单、六类、单图/校准推理 |
| EXP-01 | 两方法完整实验与效率 | A/B 各方法 | 无缺失配置、日志、真实汇总 |
| DEMO-01 | Part 1/2 演示与录屏 | A；B 提供接口 | 原图、模板/热力图、分数、阈值 |
| PRE-01 | 图表、讲稿、交叉验收 | 双方 | pre-v1、备份、双方能讲解 |

这些编号不是已创建的 GitHub Issue number。可以使用仓库 Issue 模板逐项创建；不要在文档里伪造 Issue 链接。

## 里程碑与证据

M0 协议与环境 → M1 两个单类别流程 → M2 完整实验 → M3 演示/交叉复现 → Pre 冻结。

每个 Issue 至少记录：负责人、依赖、完成条件、PR、验证命令、commit、结果位置。跑服务器的代码必须先提交或记录未提交 diff；仅有截图不足以复现。

## Pre 版本固定（共同确认后）

```bash
git switch main
git pull --ff-only origin main
git tag -a pre-v1 -m "Part 1 and Part 2 presentation snapshot"
git push origin pre-v1
```

录屏等大文件通过 GitHub Release 附件或课程指定渠道提供，README 链接到实际存在的产物。课程最终要求公开仓库，提交前检查公开可访问与 README 完整性。
