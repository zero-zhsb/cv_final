"""Compare A on the five previously selected B difficult cases without tuning."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/figures/comparison_cases')
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.data.visa import VisAReader
    from src.part1.inspector import TemplateInspector
    manifest=json.loads((ROOT/'manifests/visa_1cls.json').read_text())
    reader=VisAReader(args.source,manifest)
    selection=json.loads((ROOT/'results/figures/part2_gpu_failures/cases.json').read_text())
    if args.output.exists():
        parser.error('Output exists; preserve existing visualizations')
    args.output.mkdir(parents=True)
    rows=[]
    for index,case in enumerate(selection['cases'],1):
        category=case['category'];entry=manifest['categories'][category]
        inspector=TemplateInspector()
        inspector.fit_references([reader.path(p) for p in entry['references']['8']['0']])
        threshold=float(np.quantile([inspector.predict(reader.path(p),False)['score'] for p in entry['calibration']],.95,method='linear'))
        prediction=inspector.predict(reader.path(case['image_path']))
        sample=next(r for r in entry['tests'] if r['image_path']==case['image_path'])
        truth=reader.mask(sample,prediction['anomaly_map'].shape)
        with Image.open(reader.path(case['image_path'])) as image:
            rgb=np.asarray(image.convert('RGB'))
        row={'category':category,'shot':8,'seed':0,'image_path':case['image_path'],'label':sample['label'],
             'part1_score':prediction['score'],'part1_threshold':threshold,
             'part1_decision':'NG' if prediction['score']>threshold else 'OK',
             'part2_score':float(case['score']),'part2_threshold':float(case['threshold']),
             'part2_decision':case['decision'],'manifest_id':manifest['manifest_id']}
        rows.append(row)
        fig=plt.figure(figsize=(13,8))
        grid=fig.add_gridspec(2,4,height_ratios=[1,1.15])
        axes=[fig.add_subplot(grid[0,i]) for i in range(4)]
        axes[0].imshow(rgb);axes[0].set_title('Same input')
        axes[1].imshow(truth,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Ground truth (evaluation only)')
        plot=axes[2].imshow(prediction['anomaly_map'],cmap='inferno',vmin=0);axes[2].set_title('Part 1 heatmap')
        fig.colorbar(plot,ax=axes[2],fraction=.046)
        axes[3].imshow(rgb);axes[3].imshow(prediction['anomaly_map'],cmap='inferno',alpha=.45,vmin=0)
        axes[3].set_title(f"Part 1 {row['part1_decision']} | score={row['part1_score']:.4f}\nthreshold={threshold:.4f}")
        for axis in axes:axis.axis('off')
        bottom=fig.add_subplot(grid[1,:]);bottom.imshow(plt.imread(ROOT/'results/figures/part2_gpu_failures'/case['figure']));bottom.axis('off')
        bottom.set_title('Part 2: published genuine result REPLAY | independent color scale',fontsize=11)
        fig.suptitle(f"{case['image_path']} | 8-shot / seed 0 | true label: {'NG' if sample['label'] else 'OK'}",fontsize=13)
        fig.text(.5,.01,'Fixed, previously selected B error cases; not random samples. No parameter changes. Maps/scales differ by method.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.025,1,.96));fig.savefig(args.output/f'case_{index}_{category}.png',dpi=150);plt.close(fig)
    with (args.output/'cases.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'provenance.json').write_text(json.dumps({'selection':selection['selection'],'manifest_id':manifest['manifest_id'],
        'part1':'recomputed unchanged template method','part2':'published B visualization replay; no model rerun','count':len(rows)},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
