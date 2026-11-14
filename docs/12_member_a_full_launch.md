# A 的全量实验与演示：已启动，尚未验收完成

**后续状态更新：** 用户再次要求查询时，已确认正式运行 72/72、`run.json` 为 `completed`、日志为 `MATRIX_COMPLETED 72`，并执行完整汇总和身份核验。结果见 [全量结果说明](13_member_a_full_results.md)。以下是启动时记录，不代表当前全量仍未完成；Demo 的完整浏览器/上传验收仍待完成。

## 本轮范围

按用户授权推进 Part 1 全量、与 Part 2 的对比、实验可视化及可运行 Demo。用户随后明确：不做视频相关工作；任务 launch 完即可，不持续监控。因此本文件只记录启动状态，不声称 72 配置及全部汇总已完成。Part 3 和最终报告不在本轮范围。

已整合 `codex/midterm-video-materials` 的既有资料及可编辑 PPT，保留队友内容；不新增视频制作任务。

## 实验方法与资源

- Part 1 算法不变：短边 448，裁为 14 的倍数；最近 RGB 模板，sigma=1 平滑，最高 1% 聚合；正常校准图 95% 分位阈值。
- 六类、1/2/4/8-shot、seed 0/1/2，目标 72 配置、11,520 条测试预测。
- `VisAReader` 将官方 CSV 映射到既有 manifest；核验 train/test/mask 身份后只读访问共享 VisA，不复制整份数据，不改公共目录。
- 因数据盘接近满载，不保存全部测试异常图。推理连续分数在第一遍产生，第二遍确定性重建地图并核对分数，进行原图尺寸精确 AUROC。代表性图单独生成。
- 模板推理仍为 CPU，2 个计算线程，低优先级。可选 CuPy 仅加速像素评价中的整数计数，计数区间受约 3 GiB 显存上限约束；不能把评价显存说成模型推理显存。
- 使用 float32 原始位模式进行精确计数，不量化、不降采样、不逐图归一化。新 CPU 和 CUDA 评价均在 cashew/1-shot/seed=0 复现原结果：image AUROC 0.6534、pixel AUROC 0.7780427208851487。
- 新检查共 24 项，20 项通过，4 项 PyTorch/FAISS 模型或其 GPU 环境相关检查跳过。`CV_TEST_CUPY=1` 显式启用 CuPy 计数核验。

## 已启动的服务器任务

工作目录 `~/cv_final`：

```bash
# 已在后台启动；不要不加检查就重复启动。
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nohup nice -n 10 \
  .venv-a/bin/python scripts/run_part1_matrix.py \
  --source /data/disk1/shared/aiaa3201/project2_anomaly/VisA \
  --output runs/part1_full_stream_cuda_v1 --pixel-backend cuda \
  > logs/part1_full_stream_cuda_v1.log 2>&1 &
```

正式启动时 PID 为 `4002693`。PID 仅用于本次运行，不可据此在将来停止进程；应先核对属主和完整命令。早期 CPU 运行 `part1_full_stream_v1` 已由助手停止，原检查点保留，正式汇总不混入该运行。新运行支持在相同源码、manifest 和参数下跳过已完成配置，但不跳过尚未完成的配置。

用户之后需要查看时手动运行：

```bash
cat runs/part1_full_stream_cuda_v1/status.json
tail -20 logs/part1_full_stream_cuda_v1.log
```

只有 `run.json` 为 `completed` 且全部 72 个 `metrics.json` 存在，才运行正式汇总：

```bash
.venv-a/bin/python scripts/summarize_part1_matrix.py \
  --run runs/part1_full_stream_cuda_v1
```

汇总代码拒绝不完整矩阵，并核对与 B 的 11,520 条 image/mask/label/manifest 身份完全一致。产物为 `results/tables/part1_full/` 和 `results/figures/part1_full/`。先逐 seed 对六类别等权平均，再报告三个 seed 的均值和总体标准差。CPU Part 1 与历史 GPU Part 2 的延迟只作描述，不构造受控速度比。

## 可运行 Demo

已启动于服务器 `127.0.0.1:8765`（启动时 PID `4002694`），仅本机监听。样例 API 已验证：cashew/bad/000.JPG 的 Part 1 实时 score=0.49833202362060547、OK；B 保存结果为 NG。响应包含原图、热图和叠加图。

```bash
.venv-a/bin/python scripts/serve_demo.py \
  --source /data/disk1/shared/aiaa3201/project2_anomaly/VisA --port 8765
```

通过 VS Code 的端口转发面板转发 8765，再打开对应本机地址。或在本机终端建立 SSH 隧道：

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 -p 10084 zzhang657@121.12.5.46
```

浏览器选择类别、shot、seed 和已知测试图，也可上传相同拍摄规格的产品图。Part 1 实时输出 score、OK/NG、热图；Part 2 明确标注为保存分数回放。任意上传图片没有对应的 Part 2 历史结果，不能冒充实时 DINO。

API 已核验；完整浏览器视觉验收尚未完成。浏览器操作因会话绑定问题停止，未将该问题算作功能通过。

## 困难案例和剩余验收

`scripts/render_comparison_cases.py` 已在服务器生成 B 先前选定的五个困难案例的 A/B 对照（输出 `results/figures/comparison_cases/`）。A 结果实时重建，B 图为公开的真实保存输出回放，保持同类别/8-shot/seed=0，标明独立色标。选择不是随机抽样，不据此估计总体错误率。

本轮还没有完成：72 配置终态确认、全量表格/曲线生成、五张对照图的视觉验收、完整浏览器上传流程验收、最终运行结果回传与两端 Git 版本统一。后续从本页继续，不把后台启动当作交付完成。
