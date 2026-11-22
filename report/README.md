# CVPR 格式报告

[当前报告草稿](AIAA3201_Report_Draft.pdf) 以固定的 448+672 多尺度对照为主结果，覆盖模板基线、冻结 AnomalyDINO、分辨率消融、小缺陷同误报率对比、五个困难案例和同一会话的 GPU 效率测量。报告作者栏暂用课程组标识。提交前填入两位组员姓名，并根据团队最终选择更新摘要中的版本链接。

`main.tex` 和 `references.bib` 是可编辑源稿。使用 [CVPR 官方 author kit](https://github.com/cvpr-org/author-kit) 的提交 `a2369342011145be8c85e5ad21a9b61f9fa53473`。保留 author kit 在独立目录。从本目录用 Tectonic 构建。

```bash
tectonic -Z "search-path=/path/to/author-kit" main.tex
```

若使用 TeX Live，令 `TEXINPUTS` 和 `BSTINPUTS` 包含该目录，再以 `latexmk -pdf main.tex` 编译。报告图中的 VisA 原图、标注、叠图和裁剪注明 Zou 等人的数据来源和 CC BY 4.0 许可。流水线与指标曲线可从项目根目录运行 `python scripts/build_report_plots.py` 复现。
