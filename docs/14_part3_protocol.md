# Part 3：冻结 DINOv2 的多层特征融合

方案于 2026-11-21 在正式测试前固定，机器可读设置见 `configs/part3_protocol.json`。当前仅声明实验设计，不预先声称提高指标。

## 假设和单一改进

既有真实案例显示细小缺陷漏检。固定输入短边 448，保持 ViT-S/14 冻结，比较第 6 层局部信息和第 12 层表示的互补性。多层融合不改变 patch 网格，也不保证能恢复 patch 内细节。

三条路径为原第 12 层、单第 6 层、等权 6+12 融合。所有路径都提取第 12 层来产生完全相同的官方 PCA 前景 mask，保持 reference 旋转、归一化、精确 k=1 近邻、图像最高 1% 聚合和热图渲染不变。每个层的向量先 L2 归一化，再按 sqrt(0.5) 拼接，继续走原归一化/精确搜索。

融合匹配一个共同 reference patch，距离是两个层对该 patch 的距离均值。它与“两个层独立找最近邻后平均最小距离”不同，相关单元测试会拒绝这种偷换。层和权重不按测试 AUROC 搜索，20 张正常保留图只计算演示阈值，不拟合融合模型。

## 核验和执行

`scripts/check_part3.py` 将新增第 12 层路径与未修改的 Part 2 逐图对照，门槛为 score 1e-6、patch 2e-6，并比较三个路径的实际 PCA mask。通过后才开始全矩阵。原完整 Part 2 的 run、预测与指标保留；新代码不修改 `src/part2/inspector.py`。

`scripts/run_part3.py` 遵循同一 manifest、六类、1/2/4/8-shot、seed 0/1/2。每个变体写入独立新 run，不混拼结果、不覆盖旧实验；使用紧凑 patch map 保存，原分辨率精确 pixel AUROC 复用现有评价器。每配置保存源代码 hash、参考图、校准、阈值、资源/负载及完整预测。低于 1 GiB 可用空间会停止我们自己的新任务。

正式命令在 B 工作目录执行，先核验：

```bash
.venv-b/bin/python scripts/check_part3.py
```

GPU 推理单个进程、两个数值线程、nice=10、PyTorch 分配器上限 15%；评价最多一个进程、上限 10%。这些上限不代表预留显存。共享服务器不停止或修改其他用户任务。

```bash
nice -n 10 .venv-b/bin/python scripts/run_part3.py --variant fusion6_12 \
  --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls \
  --allow-image-root /data/disk1/shared/aiaa3201/project2_anomaly/VisA \
  --shots 1 2 4 8 --seeds 0 1 2 --device cuda:0 --threads 2 \
  --map-storage patch --knn-backend torch-gpu --warmup-iters 25 \
  --gpu-memory-fraction 0.15 --output runs/part3_fusion6_12_v1
```

单第 6 层使用同一命令，variant 改为 layer6，输出改为 `runs/part3_layer6_v1`。第 12 层控制先做真实图数值核验；原 72 配置已完整保存，可作为不变检测/定位参照。正式效率结论需在本次运行条件下另测原控制，不把历史共享负载下的延迟差直接称为算法速度比。

## 报告原则

每 shot 先对六类等权 macro 平均，再报告三个 seed 的 mean±population std。主指标 image AUROC，同时报告 pixel AUROC、实际困难案例、背景误报、平均推理时间及峰值 PyTorch 显存。无测试逐图归一化，不改变掩码或测试分辨率来降低评价成本。提升、持平和下降都如实报告；根据受控消融讨论原因，不根据测试成绩不断更换层或权重。

该实验属于课程 Part 3 Direction A。方法源于官方 AnomalyDINO / DINOv2，多层表示是课程允许的探索，不包装为已证实的原创算法。视频与报告只有在正式结果完成后才更新为实测改进。

## 本次实际运行与后处理

2026-11-21 已通过真实图 gate：三张 cashew 图像的新第 12 层路径与原 Part 2 的 score 和 patch 最大绝对误差均为 0；三个变体的最终层 PCA mask 相同。记录见 `results/tables/part3_validation.json`。这只是实现核验，不是全矩阵改进证据。

一键控制器 `scripts/run_part3_study.py` 顺序执行融合、单第 6 层各 72 配置；每条路径边推理边用一个进程做原分辨率精确 pixel AUROC。控制器持文件锁，拒绝覆盖既有不完整 run。服务器 B 目录：`/data/disk1/home/sshen560/cv_final_member_b`（服务器可能将其解析为 disk2 路径）。

进度与日志：

```bash
cat runs/part3_study_status.json
cat runs/part3_fusion6_12_v1/evaluation_status.json
tail -n 20 logs/part3_fusion6_12_inference.log
tail -n 20 logs/part3_fusion6_12_evaluation.log
```

原始结果分别在 `runs/part3_fusion6_12_v1`、`runs/part3_layer6_v1`，其中包含 `run.json`、逐图 `predictions.csv`、参考/校准/阈值、紧凑浮点 patch map 与 `metrics/*.json`。原 Part 2 仍在 `runs/part2_gpu_exact_matrix_v1`。

`scripts/finalize_part3.py` 是本次实验的一次性后处理：等待两条完整路径及评价结束后，检查各方法的全部测试身份、参考/校准名单、权重及输入等设置，生成按 shot/类别/seed 的表格、消融曲线、五个真实错误的 Part 2/Part 3 同图对照，再运行同会话效率抽样。最终状态是 `artifacts_ready`，表示材料可供审核；仍需人工查看图像、解释正负变化并提交结果，不能把它等同于报告已完成。

效率抽样固定六类、1/8-shot、seed=0、每类两正常两异常图、三次重复。三个方法的顺序按类别/shot 循环轮换，各次只驻留一个方法。共享 GPU 没有独占预留，即便同一会话也不能消除其他负载影响，结果会明确标注。模型层、融合权重和准确率协议不受计时结果调整。

预期生成文件（运行结束前可能尚不存在）：`docs/15_part3_results.md`、`docs/16_part3_failure_analysis.md`、`docs/17_part3_efficiency.md`、`results/tables/part3/` 和 `results/figures/part3/`。不把未生成或未审阅的材料提前算作完成。
