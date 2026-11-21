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
