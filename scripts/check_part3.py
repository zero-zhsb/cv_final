"""Gate the new extraction path against the unchanged Part 2 implementation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.part2.inspector import AnomalyDINOInspector
from src.part3.inspector import MultilayerInspector


def main():
    torch.cuda.set_per_process_memory_fraction(.15,0)
    data=Path('/data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls')
    manifest=json.loads((ROOT/'manifests/visa_1cls.json').read_text())
    refs=[data/p for p in manifest['categories']['cashew']['references']['1']['0']]
    images=['cashew/test/good/006.JPG','cashew/test/bad/000.JPG','cashew/test/bad/075.JPG']
    args=(ROOT/'third_party/AnomalyDINO',ROOT/'third_party/dinov2',ROOT/'weights/dinov2_vits14_pretrain.pth','cuda:0')
    baseline=AnomalyDINOInspector(*args,knn_backend='torch-gpu');baseline.fit_references(refs,0)
    original=[baseline.predict(data/p,False) for p in images]
    del baseline;torch.cuda.empty_cache()
    new=MultilayerInspector(*args,variant='layer12',knn_backend='torch-gpu');new.fit_references(refs,0)
    errors=[]
    for p,a in zip(images,original):
        b=new.predict(data/p,False)
        errors.append({'image':p,'score_error':abs(a['score']-b['score']),'patch_max_error':float(np.max(np.abs(a['patch_map']-b['patch_map'])))})
    assert max(e['score_error'] for e in errors)<1e-6,errors
    assert max(e['patch_max_error'] for e in errors)<2e-6,errors
    # Both ablations must apply the exact same final-layer PCA mask/RNG stream.
    checks={}
    for variant in ['layer12','layer6','fusion6_12']:
        del new;torch.cuda.empty_cache()
        new=MultilayerInspector(*args,variant=variant,knn_backend='torch-gpu');new.fit_references(refs,0)
        np.random.seed(43)
        masks=[]
        for p in images:
            tensor,grid=new.backbone.prepare_image(str(data/p));f=new.backbone.extract_features(tensor)
            masks.append(new.backbone.compute_background_mask(f,grid,threshold=10,masking_type=True).tolist())
        checks[variant]=masks
    assert checks['layer12']==checks['layer6']==checks['fusion6_12']
    result={'passed':True,'baseline_path':'unchanged src/part2/inspector.py','errors':errors,'identical_final_layer_masks':True,'images':images}
    out=ROOT/'results/tables/part3_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
