"""Render 1080p midterm visual assets from genuine saved scientific arrays."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.part2.maps import load_anomaly_map

OUT = ROOT / 'presentation/midterm_video'
RAW = OUT / 'raw'
ASSETS = OUT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18})
BG, FG, MUTED, BLUE, TEAL = '#101a2d', '#f5f7fc', '#b9c7dd', '#69aaff', '#56d6b6'
SELECTION = {x['name']: x for x in json.loads((RAW / 'selection.json').read_text())}


def canvas(number, heading, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    fig.text(.045, .94, heading, fontsize=29, weight='bold', color=FG)
    fig.text(.045, .885, subtitle, fontsize=15, color=MUTED)
    fig.text(.045, .035, 'MIDTERM CHECKPOINT  |  VisA  |  Genuine saved experimental outputs', fontsize=10, color=MUTED)
    fig.text(.95, .035, f'{number:02}', ha='right', fontsize=12, color=MUTED)
    return fig


def text(fig, x, y, value, size=20, color=FG, **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, **kwargs)


def rgb(name):
    return np.asarray(Image.open(RAW / name).convert('RGB'))


def mask(name):
    return np.asarray(Image.open(RAW / name).convert('L')) > 0


def panel(fig, rect, data, title, heat=False, overlay=None):
    ax = fig.add_axes(rect)
    if heat:
        im = ax.imshow(data, cmap='inferno', vmin=0)
        cax = fig.add_axes([rect[0] + rect[2] + .004, rect[1] + .035, .006, rect[3] - .07])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9, colors=MUTED)
    else:
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None, vmin=0 if data.ndim == 2 else None,
                  vmax=1 if data.ndim == 2 else None)
    if overlay is not None:
        ax.imshow(overlay, cmap='inferno', alpha=.45, vmin=0)
    ax.axis('off');ax.set_title(title, fontsize=16, color=FG, pad=12)
    return ax


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=120, facecolor=BG)
    plt.close(fig)


def decision_line(fig, row, outcome):
    text(fig, .055, .16, f"Score {float(row['score']):.4f}    |    Threshold {float(row['threshold']):.4f}    |    Output: {row['decision']}", 23)
    text(fig, .055, .105, outcome, 17, color=TEAL)


def main():
    fig=canvas(1, 'Few-shot industrial anomaly detection', 'Can a few normal examples reveal an unseen product defect?')
    panel(fig,[.055,.23,.4,.55],rgb('normal.JPG'),'Normal product')
    ax=panel(fig,[.545,.23,.4,.55],rgb('defect.JPG'),'Defective product + ground-truth outline')
    ax.contour(mask('defect_mask.png'),levels=[.5],colors=['#ffdd67'],linewidths=1.5)
    text(fig,.055,.13,'Input: a few normal references + a new product image',22)
    text(fig,.055,.08,'Output: anomaly location, continuous score, and calibrated OK / NG',19,color=MUTED)
    save(fig,'01_problem.png')

    fig=canvas(2,'Our current technical approach','A simple RGB baseline, followed by a frozen-feature method')
    def flow(y,label,items,color):
        text(fig,.055,y+.14,label,21,color=color,weight='bold')
        for i,item in enumerate(items):
            x=.055+i*.235
            fig.patches.append(FancyBboxPatch((x,y),.195,.105,boxstyle='round,pad=0.012',transform=fig.transFigure,
                                            facecolor='#1e304e',edgecolor=color,linewidth=1.5))
            text(fig,x+.0975,y+.052,item,17,ha='center',va='center')
            if i<3:
                fig.patches.append(FancyArrowPatch((x+.205,y+.05),(x+.225,y+.05),arrowstyle='-|>',mutation_scale=18,
                                                 transform=fig.transFigure,color=MUTED))
    flow(.57,'PART 1  |  Template difference',['Normal templates','Nearest template','RGB difference','Map + image score'],BLUE)
    flow(.31,'PART 2  |  AnomalyDINO',['Frozen DINOv2','Normal patch bank','Exact nearest neighbor','Map + image score'],TEAL)
    text(fig,.055,.19,'Shared reference/test manifest  |  No defect-label training',21)
    text(fig,.055,.115,'20 separate normal calibration images per category set the OK / NG threshold.',17,color=MUTED)
    text(fig,.055,.075,'Calibration images are excluded from the reference bank and test set.',16,color=MUTED)
    save(fig,'02_method.png')

    positions=[[.035,.32,.205,.42],[.275,.32,.205,.42],[.515,.32,.205,.42],[.765,.32,.205,.42]]
    fig=canvas(3,'Part 1: a working template-difference baseline','cashew/test/bad/000.JPG  |  1-shot, seed 0  |  Saved output from actual inference')
    amap=load_anomaly_map(RAW/'defect_part1.npz')
    for rect,data,title,heat,over in zip(positions,[rgb('defect.JPG'),rgb('reference.JPG'),amap,rgb('defect.JPG')],
           ['Product input','Normal template','Raw anomaly map','Overlay'],[False,False,True,False],[None,None,None,amap]):
        panel(fig,rect,data,title,heat,over)
    decision_line(fig,SELECTION['defect']['part1'],'Ground truth: defective. The baseline misses this image (false negative).')
    save(fig,'03_part1.png')

    fig=canvas(4,'Part 2: frozen features and normal patch memory','Same image, reference, shot and seed  |  GPU primary-run output; no new inference during playback')
    amap=load_anomaly_map(RAW/'defect_part2.npz')
    for rect,data,title,heat,over in zip(positions,[rgb('defect.JPG'),mask('defect_mask.png'),amap,rgb('defect.JPG')],
           ['Product input','Ground-truth mask','Raw anomaly map','Overlay'],[False,False,True,False],[None,None,None,amap]):
        panel(fig,rect,data,title,heat,over)
    decision_line(fig,SELECTION['defect']['part2'],'Ground truth: defective. Part 2 flags this image as NG (true positive).')
    text(fig,.055,.07,'Each method has its own color scale and calibration threshold; raw scores are not directly comparable.',12,color=MUTED)
    save(fig,'04_part2.png')

    rows=list(csv.DictReader((ROOT/'results/tables/part1_part2_cashew_milestone.csv').open()))
    assert len(rows)==2 and all(int(r['test_count'])==150 for r in rows)
    fig=canvas(5,'Preliminary comparison on identical test images','cashew only  |  1-shot, seed 0  |  150 tests: 50 normal + 100 anomalous')
    ax=fig.add_axes([.08,.27,.52,.53],facecolor=BG)
    xs=np.arange(2);w=.32
    for i,(r,c,label) in enumerate(zip(rows,[BLUE,TEAL],['Part 1: RGB template','Part 2: AnomalyDINO'])):
        vals=[float(r[k])*100 for k in ['image_auroc','pixel_auroc']]
        bars=ax.bar(xs+(i-.5)*w,vals,w,color=c,label=label)
        ax.bar_label(bars,labels=[f'{v:.2f}%' for v in vals],padding=6,color=FG,fontsize=15)
    ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xticks(xs,['Image AUROC','Pixel AUROC']);ax.tick_params(colors=FG,labelsize=16)
    for s in ax.spines.values():s.set_color(MUTED)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='lower left',facecolor=BG,edgecolor=MUTED,labelcolor=FG,fontsize=12)
    text(fig,.65,.69,'Same data identities',24,weight='bold')
    text(fig,.65,.61,'Same normal references\nSame test images and masks\nSame AUROC implementation',18,linespacing=1.7,color=MUTED,va='top')
    text(fig,.65,.36,'Single configuration',24,color=TEAL,weight='bold')
    text(fig,.65,.29,'Not a six-category average.\nNo cross-seed uncertainty yet.',16,linespacing=1.5,color=MUTED)
    text(fig,.08,.13,'Feature matching performs better in this preliminary configuration.',21)
    text(fig,.08,.08,'No controlled speed comparison: the two runs used different devices/load conditions.',14,color=MUTED)
    save(fig,'05_preliminary_comparison.png')

    rows=list(csv.DictReader((ROOT/'results/tables/part2_gpu_overall_by_shot.csv').open()))
    fig=canvas(6,'Part 2: progress beyond the first configuration','Six VisA categories x 4 shot counts x 3 seeds  |  72 configurations; 11,520 test predictions')
    for idx,(metric,label,color) in enumerate([('image','Image AUROC',BLUE),('pixel','Pixel AUROC',TEAL)]):
        ax=fig.add_axes([.075+idx*.48,.31,.4,.47],facecolor=BG)
        means=np.array([float(r[metric+'_auroc_mean']) for r in rows])*100
        std=np.array([float(r[metric+'_auroc_std_ddof0']) for r in rows])*100
        ax.errorbar(range(4),means,yerr=std,fmt='o-',color=color,capsize=6,lw=3,ms=9)
        for i,v in enumerate(means):ax.annotate(f'{v:.2f}%',(i,v),xytext=(0,17),textcoords='offset points',ha='center',color=FG,fontsize=15)
        ax.set_xlim(-.25,3.25);ax.set_ylim(90,100.5);ax.set_xticks(range(4),['1','2','4','8']);ax.set_xlabel('Normal reference images',color=FG,fontsize=15)
        ax.set_title(label,color=FG,fontsize=23,pad=20);ax.yaxis.set_major_formatter(PercentFormatter(100));ax.tick_params(colors=MUTED,labelsize=13)
        ax.grid(alpha=.15)
        for s in ax.spines.values():s.set_color(MUTED)
    text(fig,.075,.16,'8-shot: image AUROC 96.62 +/- 0.57%; pixel AUROC 98.21 +/- 0.02%',22)
    text(fig,.075,.10,'Category macro average per seed, then mean +/- population standard deviation over 3 seeds.',14,color=MUTED)
    text(fig,.075,.065,'Separate experiment scope from the single-category Part 1/2 comparison on the previous card.',13,color=MUTED)
    save(fig,'06_part2_progress.png')

    fig=canvas(7,'Current limitations and concrete next steps','A genuine Part 2 failure: cashew/test/bad/075.JPG  |  8-shot, seed 0; a different configuration')
    amap=load_anomaly_map(RAW/'failure_part2.npz')
    for x,data,label,over in [(.035,rgb('failure.JPG'),'Defective input',None),(.235,mask('failure_mask.png'),'Tiny GT defects',None),(.435,rgb('failure.JPG'),'Part 2 overlay',amap)]:
        panel(fig,[x,.38,.18,.34],data,label,overlay=over)
    row=SELECTION['failure']['part2']
    text(fig,.04,.28,f"Output: OK  |  score {float(row['score']):.4f} < threshold {float(row['threshold']):.4f}",17)
    text(fig,.04,.20,'False negative; patch granularity may contribute.',16,color=MUTED)
    text(fig,.04,.15,'This explanation needs a controlled experiment.',15,color=MUTED)
    for y,title,detail in [(.73,'1  Complete baseline coverage','Run Part 1 on all six classes, shots and seeds.'),(.55,'2  Test multi-scale features','Explore a second scale for small-defect signals.'),(.37,'3  Run a controlled ablation','Fix references/seeds; compare AUROC, latency\nand memory against unchanged Part 2.')]:
        text(fig,.66,y,title,19,color=TEAL,weight='bold');text(fig,.66,y-.06,detail,14,linespacing=1.6,color=MUTED)
    text(fig,.66,.16,'Proposed work; no improvement is claimed yet.',14,color=FG)
    save(fig,'07_limitations_plan.png')

    fig=canvas(8,'Optional normal-product check','cashew/test/good/006.JPG  |  1-shot, seed 0  |  Both methods output OK')
    panel(fig,[.06,.28,.37,.5],rgb('normal.JPG'),'Normal product')
    for j,k in enumerate(['part1','part2']):
        row=SELECTION['normal'][k]
        text(fig,.51,.68-j*.22,'Part '+str(j+1)+'  |  '+row['decision'],27,color=BLUE if j==0 else TEAL)
        text(fig,.51,.61-j*.22,f"score {float(row['score']):.4f}  |  threshold {float(row['threshold']):.4f}",20,color=MUTED)
    text(fig,.06,.14,'A normal example complements the defect and failure examples.',21)
    save(fig,'08_optional_normal_check.png')

    provenance={'assets_are':'Static visualization of genuine saved outputs, not screen recordings',
                'source_branch':'codex/part1-integration','source_commit':'a3e03fb891155d5ca043b9f3bbf79144d52f4569',
                'part1_run':'part1_cashew_1shot_seed0_v1','part2_run':'part2_gpu_exact_matrix_v1',
                'manifest_id':SELECTION['defect']['part1']['manifest_id'],
                'selection':list(SELECTION.values()),
                'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file()}}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Generated',len(list(ASSETS.glob('*.png'))),'1920x1080 cards')


if __name__=='__main__':main()
