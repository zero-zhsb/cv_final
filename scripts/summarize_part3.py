"""Require complete matched studies before reporting the preregistered ablation."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def tiny_mask_paths(data, manifest, limit):
    from PIL import Image

    selected = set()
    for category in manifest["categories"].values():
        for row in category["tests"]:
            if row["label"] != 1:
                continue
            with Image.open(data / row["mask_path"]) as handle:
                mask = np.asarray(handle.convert("L")) > 0
            if 0 < mask.mean() <= limit:
                selected.add(row["mask_path"])
    return selected


def tiny_localization_recall(data, run, row):
    """Recall of GT pixels in the top 1% of the original-size map, with ties shared."""
    from PIL import Image
    from src.part2.maps import load_anomaly_map

    with Image.open(data / row["mask_path"]) as handle:
        mask = np.asarray(handle.convert("L")) > 0
    anomaly = load_anomaly_map(run / row["map_path"])
    if anomaly.shape != mask.shape:
        raise ValueError("Mask and map shapes differ: " + row["image_path"])
    count = max(1, anomaly.size // 100)
    cutoff = np.partition(anomaly.ravel(), anomaly.size - count)[-count]
    higher = anomaly > cutoff
    tied = anomaly == cutoff
    fraction = (count - higher.sum()) / tied.sum()
    true_pixels = (higher & mask).sum() + fraction * (tied & mask).sum()
    return float(true_pixels / mask.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["multilayer", "multiscale"], default="multilayer")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    multiscale = args.study == "multiscale"
    if multiscale and args.data_root is None:
        parser.error("--data-root is required for multiscale small-defect reporting")
    baseline=ROOT/'runs/part2_gpu_exact_matrix_v1'
    manifest=json.loads((ROOT/'manifests/visa_1cls.json').read_text())
    base_variant = "part2_448" if multiscale else "layer12"
    studies = {base_variant: baseline}
    variants = ["scale672", "fusion448_672"] if multiscale else ["layer6", "fusion6_12"]
    studies.update({v: ROOT / f"runs/part3_{v}_v1" for v in variants})
    report_name = "part3_multiscale" if multiscale else "part3"
    if multiscale:
        protocol = json.loads((ROOT / "configs/part3_multiscale_protocol.json").read_text())
        tiny_paths = tiny_mask_paths(args.data_root, manifest, protocol["tiny_defect_mask_fraction_max"])
        tiny_rows = []
    def identity(r):return (r['category'],r['shot'],r['seed'],r['image_path'],r['mask_path'] or '',r['label'],r['manifest_id'])
    baseline_ids={identity(r) for r in read(baseline/'predictions.csv')}
    baseline_meta=json.loads((baseline/'run.json').read_text())
    baseline_thresholds=json.loads((baseline/'thresholds.json').read_text())
    invariants=['manifest_id','weights_sha256','upstream_commit','dinov2_commit',
                'preprocess','precision','knn_backend','warmup_iters',
                'categories','shots','seeds']
    records=[];metadata={};overall=[]
    for variant,run in studies.items():
        meta=json.loads((run/'run.json').read_text());metrics=[json.loads(p.read_text()) for p in (run/'metrics').glob('*.json')]
        expected={(c,k,s) for c in manifest['categories'] for k in manifest['shots'] for s in manifest['seeds']}
        actual={(r['category'],r['shot'],r['seed']) for r in metrics}
        pred=read(run/'predictions.csv')
        if meta['status']!='completed' or len(metrics)!=72 or actual!=expected or len(pred)!=11520 or len({identity(r) for r in pred})!=11520 or {identity(r) for r in pred}!=baseline_ids:
            raise ValueError('Incomplete or mismatched study: '+variant)
        if meta.get('code_dirty') or meta.get('partial_smoke_test'):
            raise ValueError('Require clean-source full study: '+variant)
        for key in invariants:
            if meta[key]!=baseline_meta[key]:raise ValueError(f'Changed {key}: {variant}')
        expected_resolution = 672 if multiscale and variant != base_variant else 448
        if meta["resolution"] != expected_resolution:
            raise ValueError("Unexpected resolution: " + variant)
        if multiscale and variant != base_variant and meta["protocol"] != protocol:
            raise ValueError("Unexpected protocol: " + variant)
        for key in ['src/part2/inspector.py','src/part2/gpu_index.py']:
            if meta['source_sha256'][key]!=baseline_meta['source_sha256'][key]:
                raise ValueError('Changed baseline core: '+key)
        thresholds=json.loads((run/'thresholds.json').read_text())
        if set(thresholds)!=set(baseline_thresholds):raise ValueError('Mismatched threshold configurations')
        for key,t in thresholds.items():
            for field in ['references','calibration']:
                if t[field]!=baseline_thresholds[key][field]:
                    raise ValueError(f'Mismatched {field}: {variant}/{key}')
        if any(not np.isfinite(float(m[k])) for m in metrics for k in ['image_auroc','pixel_auroc','mean_inference_ms']):
            raise ValueError('Non-finite metrics: '+variant)
        metadata[variant]=meta
        for m in metrics:
            group=[r for r in pred if r['category']==m['category'] and int(r['shot'])==m['shot'] and int(r['seed'])==m['seed']]
            normal=[r for r in group if r['label']=='0'];anomaly=[r for r in group if r['label']=='1']
            records.append({'variant':variant,'category':m['category'],'shot':m['shot'],'seed':m['seed'],
                'image_auroc':m['image_auroc'],'pixel_auroc':m['pixel_auroc'],
                'normal_fpr':sum(r['decision']=='NG' for r in normal)/len(normal),
                'anomaly_fnr':sum(r['decision']=='OK' for r in anomaly)/len(anomaly),
                'mean_inference_ms':m['mean_inference_ms'],'peak_gpu_memory_mib':m['peak_gpu_memory_mib']})
            if multiscale:
                # Empirical ROC point for comparison only. The demo retains its train-only threshold.
                sorted_normal = sorted((float(r["score"]) for r in normal), reverse=True)
                cutoff = sorted_normal[int(len(normal) * protocol["matched_test_fpr"])]
                tiny = [r for r in anomaly if r["mask_path"] in tiny_paths]
                tiny_rows.append({
                    "variant": variant, "category": m["category"], "shot": m["shot"], "seed": m["seed"],
                    "tiny_count": len(tiny), "test_fpr": np.mean([float(r["score"]) > cutoff for r in normal]),
                    "all_defect_fnr_at_test_fpr_cap": np.mean([float(r["score"]) <= cutoff for r in anomaly]),
                    "tiny_fnr_at_test_fpr_cap": np.mean([float(r["score"]) <= cutoff for r in tiny]) if tiny else "",
                    "tiny_gt_recall_at_top1pct_pixels": np.mean([
                        tiny_localization_recall(args.data_root, run, r) for r in tiny
                    ]) if tiny else "",
                })
        for shot in manifest['shots']:
            row={'variant':variant,'shot':shot,'categories':6,'seeds':3}
            for metric in ['image_auroc','pixel_auroc','normal_fpr','anomaly_fnr','mean_inference_ms']:
                values=[np.mean([r[metric] for r in records if r['variant']==variant and r['shot']==shot and r['seed']==s]) for s in manifest['seeds']]
                row[metric+'_mean']=float(np.mean(values));row[metric+'_std_ddof0']=float(np.std(values))
            if variant!=base_variant:
                ref=next(r for r in overall if r['variant']==base_variant and r['shot']==shot)
                row['image_delta_to_baseline']=row['image_auroc_mean']-ref['image_auroc_mean']
                row['pixel_delta_to_baseline']=row['pixel_auroc_mean']-ref['pixel_auroc_mean']
            else:row['image_delta_to_baseline']=row['pixel_delta_to_baseline']=0.
            for metric in ['image_auroc','pixel_auroc']:
                delta=[np.mean([r[metric]-next(b[metric] for b in records if b['variant']==base_variant and b['category']==r['category'] and b['shot']==shot and b['seed']==seed)
                    for r in records if r['variant']==variant and r['shot']==shot and r['seed']==seed]) for seed in manifest['seeds']]
                row[metric+'_paired_delta_std_ddof0']=float(np.std(delta))
            overall.append(row)
    out=ROOT/'results/tables'/report_name;write(out/'per_configuration.csv',records);write(out/'overall_by_shot.csv',overall)
    if multiscale:
        write(out / "tiny_defects_by_configuration.csv", tiny_rows)
        tiny_summary = []
        for variant in studies:
            for shot in manifest["shots"]:
                subset = [r for r in tiny_rows if r["variant"] == variant and r["shot"] == shot]
                row = {"variant": variant, "shot": shot,
                       "tiny_images": sum(r["tiny_count"] for r in subset if r["seed"] == 0),
                       "categories_with_tiny_images": sum(bool(r["tiny_count"]) for r in subset if r["seed"] == 0)}
                for key in ("test_fpr", "all_defect_fnr_at_test_fpr_cap", "tiny_fnr_at_test_fpr_cap",
                            "tiny_gt_recall_at_top1pct_pixels"):
                    values = [r[key] for r in subset if r[key] != ""]
                    row[key] = float(np.mean(values)) if values else ""
                tiny_summary.append(row)
        write(out / "tiny_defects_by_shot.csv", tiny_summary)
    by_category=[]
    for variant in studies:
        for category in manifest['categories']:
            for shot in manifest['shots']:
                row={'variant':variant,'category':category,'shot':shot}
                for metric in ['image_auroc','pixel_auroc','normal_fpr','anomaly_fnr']:
                    values=[r[metric] for r in records if r['variant']==variant and r['category']==category and r['shot']==shot]
                    row[metric+'_mean']=float(np.mean(values));row[metric+'_std_ddof0']=float(np.std(values))
                    base=[r[metric] for r in records if r['variant']==base_variant and r['category']==category and r['shot']==shot]
                    row[metric+'_delta_to_baseline']=float(np.mean(values)-np.mean(base))
                by_category.append(row)
    write(out/'by_category_shot.csv',by_category)
    (out/'sources.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps({'passed':True,'configurations_per_variant':72,
        'prediction_identities_per_variant':11520,'matching_metadata_fields':invariants,
        'unchanged_baseline_core':True,'identical_references_and_calibration':True,
        'finite_metrics':True,'clean_source_full_runs':True,
        'input_short_edge_by_variant':{variant:meta['resolution'] for variant,meta in metadata.items()}},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['image_auroc','pixel_auroc']):
        for variant in studies:
            rows=[r for r in overall if r['variant']==variant]
            ax.errorbar([r['shot'] for r in rows],[100*r[metric+'_mean'] for r in rows],yerr=[100*r[metric+'_std_ddof0'] for r in rows],marker='o',capsize=4,label=variant)
        ax.set(xlabel='Normal reference images',ylabel='AUROC (%)',title=metric,xticks=[1,2,4,8]);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Frozen DINOv2: single-scale controls and spatial fusion' if multiscale else
                 'Fixed 448 input, frozen backbone, final-layer masks held constant')
    fig.tight_layout();dest=ROOT/'results/figures'/report_name;dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/'ablation.png',dpi=170);plt.close(fig)
    lines=['# Part 3 full ablation results','','All variants use matched references/tests. The layer-12 baseline is the preserved Part 2 run. Proposed fusion was fixed before test evaluation. No test-label layer/weight search.','',
        '| Variant | Shots | Image AUROC (%) | Pixel AUROC (%) | Image delta (pp) | Pixel delta (pp) |','|---|---:|---:|---:|---:|---:|']
    for r in overall:lines.append(f"| {r['variant']} | {r['shot']} | {100*r['image_auroc_mean']:.2f} ± {100*r['image_auroc_std_ddof0']:.2f} | {100*r['pixel_auroc_mean']:.2f} ± {100*r['pixel_auroc_std_ddof0']:.2f} | {100*r['image_delta_to_baseline']:+.2f} | {100*r['pixel_delta_to_baseline']:+.2f} |")
    lines+=['','Mean is category macro per seed, then mean/population SD over three seeds. FPR/FNR in the CSV use the same category-macro convention.','',
        'Per-category mean/SD and signed deltas are in `results/tables/part3/by_category_shot.csv`. Paired seed macro delta SD is in `overall_by_shot.csv`; this is not a confidence interval or a significance test.','',
        'Latency values use category macro for description only. The baseline was measured on a different shared-GPU load/date, so this table does not establish a controlled speed ratio. Peak PyTorch inference allocations are per configuration in the CSV; evaluator memory is separate.','',
        'Fusion does not increase spatial patch resolution. Positive and negative changes must both be reported. Use paired category/seed results and genuine failure visualizations before attributing causes.']
    if multiscale:
        lines[0] = "# Part 3 multiscale ablation"
        lines[2] = ("The frozen Part 2 at 448 is the baseline. The 672 and 448+672 variants share a 672 PCA mask. "
                    "The previous study motivated the hypothesis. Scales and weights were fixed before the new outputs.")
        lines = [line.replace("results/tables/part3/", "results/tables/part3_multiscale/") for line in lines]
        lines[-1] = ("The tiny subset uses nonempty masks covering at most 0.1% of original image pixels. "
                     "See `tiny_defects_by_shot.csv` for category-macro recall of tiny GT pixels in the top 1% "
                     "of map pixels and the miss rate with an empirical test FPR capped at 5%. "
                     "The FPR-capped analysis compares operating points. The OK/NG demo threshold is fit on normal calibration data.")
        lines.extend(["", "![Ablation curves](../results/figures/part3_multiscale/ablation.png)"])
    doc = '19_part3_multiscale_results.md' if multiscale else '15_part3_results.md'
    (ROOT/'docs'/doc).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':True,'configurations':len(records),'overall':overall}),flush=True)


if __name__=='__main__':main()
