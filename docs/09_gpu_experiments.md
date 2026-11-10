# Part 2 GPU 实验与结果说明

日期：2026-11-10。用户已确认可以在需要时使用服务器 GPU；按共享设备运行，单个 GPU 推理进程、两个 CPU 数值线程、低 CPU 调度优先级，不停止他人任务。PyTorch 分配器上限为该 GPU 显存的 15%（上限不是预留显存）。

## 第一组完整预检

cashew，1-shot，seed=0，正常测试 50 张、异常测试 100 张：image AUROC **0.929**，pixel AUROC **0.9869356902**。仅该类别该配置，不代表六类或三个 seed 的平均成绩。精确像素评估进程峰值 RSS 约 3.03 GiB。设备上的观测平均推理时间约 219.3 ms/图，不能解释为独占硬件速度。

## 固定协议

- 六类 VisA，1/2/4/8-shot，seed 0/1/2，共 72 个类别配置、11,520 张配置内测试预测。
- 完全复用已提交的 manifest，每配置额外 20 张正常校准图不计入测试，也不加入 reference。
- 固定官方 AnomalyDINO 与 DINOv2 来源；float32，resolution=448，agnostic，CPU FAISS，无特征缓存。
- 正式运行使用 `--map-storage patch`：保存原始 float32 patch 图和原图尺寸。加载时先转换回上游使用的 float64，按官方 resize + sigma=4 平滑，再转 float32，**不下采样评估图、不逐图归一化**。
- 在 5 张真实 cashew 已保存完整图上验证，compact 重建与原导出 float32 图逐像素完全一致；另有数值与 roundtrip 测试。
- 每配置完整性检查通过后才计算 AUROC，正常测试图也参加像素评价。图像/像素 AUROC 按类别计算；最终先按 seed 对六类等权平均，再对三个 seed 算 mean±std（ddof=0）。

## 运行

服务器 B 工作目录及环境见 [工作记录](08_member_b_progress.md)。完整矩阵命令（输出目录必须不存在）：

```bash
source .venv-b/bin/activate
nice -n 10 python scripts/run_part2.py --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --allow-image-root /data/disk1/shared/aiaa3201/project2_anomaly/VisA --shots 1 2 4 8 --seeds 0 1 2 --device cuda:0 --threads 2 --map-storage patch --warmup-iters 25 --gpu-memory-fraction 0.15 --output runs/part2_gpu_matrix_v1
```

在另一个远程终端或后台进程中，串行评价已完成的配置：

```bash
nice -n 19 python scripts/watch_part2_evaluation.py --run runs/part2_gpu_matrix_v1 --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --memory-budget-gib 8
```

这只是本项目进程间的串行工作安排，不是服务器正式排队系统。每个完成配置会原子更新 run.json、阈值和校准记录；完整 CSV 和原始 patch 图在服务器，GitHub 保存统计、来源、代码和代表图片。

## 精确像素 AUROC 的内存处理

CPU 版 `src/part2/metrics.py` 汇总每幅图的独特 float32 分数及正/负像素计数，按层分批合并后计算完整排序概率，平分同分值。它没有对分数分箱或近似量化；与 sklearn AUROC 在含同分值的多图测试上匹配，默认估算工作内存预算 8 GiB，超限报错而不是静默降采样。

矩阵实际评价改用 `--pixel-device cuda:0`：`gpu_metrics.py` 利用非负有限 float32 位模式的顺序，按每个**精确值**计数，以 int32 累加，不做近似分箱；零值另行计数以避免巨大空区间。评估进程的 PyTorch 显存上限为设备的 10%，与单一推理进程的 15% 上限分别记录。此阶段只有一个推理进程，另有一个显存受限的评估进程。

已在 5 个完整 candle 配置上将 GPU 与 CPU 精确像素 AUROC 交叉核对，差值均为 0；另有两项显式启用的 GPU 数值测试。candle 约 3 亿像素的 GPU 评估占约 3.3 GiB 显存、0.8 GiB 主存。GPU 后端仅加速指标计数，不改变模型、原始异常图或分数；超出显存范围会报错，不会自动降低分辨率。

完整命令是在上述 watcher 命令末尾加入 `--pixel-device cuda:0`。普通单元测试不默认使用 GPU，只有设置 `CV_TEST_CUDA=1` 才运行 GPU 检查。

## 完成后汇总与展示

```bash
python scripts/summarize_part2.py --run runs/part2_gpu_matrix_v1 --output results/tables --figures results/figures/part2_gpu
python scripts/analyze_part2_failures.py --run runs/part2_gpu_matrix_v1 --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --output results/figures/part2_gpu_failures --shot 8 --seed 0 --count 5
```

这两个程序要求推理和相应完整结果已完成，拒绝把小样例当完整成绩。汇总提供 72 配置、24 类别/shot 与 4 个整体 shot 结果、原始预测 CSV、曲线。困难案例从固定 8-shot/seed=0 的真实阈值错误中按类别分散选择，不修改方法参数。

单图演示入口 `scripts/predict_part2.py` 读取已校准配置，只接收产品图片，不需要真值；输出 score、OK/NG、原始异常图和可视化。A 可调用该接口制作正式 demo，不能把 Part 2 接口当成已经完成的 Part 1。

## 效率解释

- 每图 inference_ms 包括图像读取、几何预处理、DINOv2、PCA mask、CPU FAISS，CUDA 前后同步；不包括热力图重建和文件导出。该范围与上游计时边界保持一致。
- 每配置单独记录 reference bank 建立时间、峰值 PyTorch allocated GPU 显存。设备 context、CUDA 库和其他进程显存不在该数字中。
- 共享设备下的时间是当次观察值，不能当成独占 A6000 的性能保证。记录外部负载，正式对比 Part 1 时使用同设备、同计时定义并披露运行条件。
- 完整矩阵实际进度与最终结果另附；启动任务不等于实验已完成。
