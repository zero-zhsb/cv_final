# 共享数据名单

在真实 VisA `1cls` 数据准备好后运行：

```bash
python scripts/prepare_manifests.py --data-root /srv/cv-final/data/VisA_pytorch/1cls --output manifests/visa_1cls.json
```

此目录可以提交生成的 JSON，不提交原图。A/B 使用同一 manifest_id，路径相对数据根。`visa_1cls.json` 已从服务器课程预置的真实 VisA、官方 1cls 划分生成；具体路径和计数见成员 B 工作记录。单元测试中的合成文件名不属于该真实名单。

当前服务器使用轻量符号链接视图。重新生成时需要加 `--allow-image-root /data/disk1/shared/aiaa3201/project2_anomaly/VisA`；只有明确授权的课程原图根可以越出 data-root。普通完整复制的数据目录不需要该参数。
