# 共享数据名单

在真实 VisA `1cls` 数据准备好后运行：

```bash
python scripts/prepare_manifests.py --data-root /srv/cv-final/data/VisA_pytorch/1cls --output manifests/visa_1cls.json
```

此目录可以提交生成的 JSON，不提交原图。A/B 使用同一 manifest_id，路径相对数据根。当前没有真实数据名单；测试中的合成文件名不是实验数据。
