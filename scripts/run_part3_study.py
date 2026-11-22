"""Bounded unattended execution of the preregistered Part 3 study, never edits other jobs."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,fcntl
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["multilayer", "multiscale"], default="multilayer")
    multiscale = parser.parse_args().study == "multiscale"
    study_name = "part3_multiscale" if multiscale else "part3_study"
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    lock=(ROOT/f'runs/{study_name}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    variants = ["scale672", "fusion448_672"] if multiscale else ["fusion6_12", "layer6"]
    if multiscale:
        gate = json.loads((ROOT / "runs/part3_multiscale_gate.json").read_text())
        protocol = (ROOT / "configs/part3_multiscale_protocol.json").read_bytes()
        if not (gate["passed"] and gate["normal_only"] and gate["matching_fine_masks"]
                and gate["protocol_sha256"] == hashlib.sha256(protocol).hexdigest()):
            raise ValueError("Run the matching normal-only multiscale gate first")
    else:
        gate=json.loads((ROOT/'results/tables/part3_validation.json').read_text())
        if not (gate['passed'] and gate['identical_final_layer_masks']):
            raise ValueError("Multilayer gate failed")
    status=ROOT/f'runs/{study_name}_status.json'
    def save(stage,**kw):
        p=status.with_suffix('.tmp');p.write_text(json.dumps({'stage':stage,'time_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n');p.replace(status)
    data='/data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls'
    try:
        for variant in variants:
            run=ROOT/f'runs/part3_{variant}_v1';save('running',variant=variant)
            with (ROOT/f'logs/part3_{variant}_inference.log').open('a') as log:
                if run.exists():
                    meta=json.loads((run/'run.json').read_text())
                    if meta['status']!='completed':raise RuntimeError('Preserving incomplete run; inspect before restart: '+str(run))
                    inference=None
                else:
                    inference=subprocess.Popen(['nice','-n','10',sys.executable,'scripts/run_part3.py','--variant',variant,'--data-root',data,'--allow-image-root','/data/disk1/shared/aiaa3201/project2_anomaly/VisA','--shots','1','2','4','8','--seeds','0','1','2','--device','cuda:0','--threads','2','--map-storage','patch','--knn-backend','torch-gpu','--warmup-iters','25','--gpu-memory-fraction','0.15','--output',str(run)],cwd=ROOT,stdout=log,stderr=log)
                # Wait only for metadata initialization, with a bounded early-exit check.
                import time
                for _ in range(120):
                    if (run/'run.json').exists():break
                    if inference is not None and inference.poll() is not None:raise RuntimeError('Inference failed before initialization')
                    time.sleep(1)
                else:raise RuntimeError('Inference initialization timed out')
                with (ROOT/f'logs/part3_{variant}_evaluation.log').open('a') as evaluation_log:
                    evaluator=subprocess.Popen(['nice','-n','19',sys.executable,'scripts/watch_part2_parallel_evaluation.py','--run',str(run),'--data-root',data,'--workers','1','--pixel-device','cuda:0'],cwd=ROOT,stdout=evaluation_log,stderr=evaluation_log)
                    code=0 if inference is None else inference.wait();eval_code=evaluator.wait()
                    if code or eval_code:raise RuntimeError(f'{variant} inference={code}, evaluation={eval_code}')
            save('variant_completed',variant=variant)
        save('summarizing')
        if multiscale:
            subprocess.run([sys.executable, 'scripts/summarize_part3.py', '--study', 'multiscale',
                            '--data-root', data], cwd=ROOT, check=True)
            subprocess.run([sys.executable, 'scripts/analyze_part2_failures.py',
                            '--run', str(ROOT / 'runs/part3_fusion448_672_v1'), '--data-root', data,
                            '--output', str(ROOT / 'results/figures/part3_multiscale/fusion_failures'),
                            '--compare-run', 'Part 2 (448)=' + str(ROOT / 'runs/part2_gpu_exact_matrix_v1'),
                            '--compare-run', 'Single scale (672)=' + str(ROOT / 'runs/part3_scale672_v1'),
                            '--method-label', 'Fusion (448+672)',
                            '--shot', '8', '--seed', '0', '--count', '5'], cwd=ROOT, check=True)
            save('benchmarking')
            subprocess.run(['nice', '-n', '10', sys.executable, 'scripts/benchmark_part3.py',
                            '--study', 'multiscale', '--data-root', data], cwd=ROOT, check=True)
            save('artifacts_ready', required_artifacts=['results/tables/part3_multiscale',
                 'docs/19_part3_multiscale_results.md', 'docs/20_part3_multiscale_efficiency.md',
                 'results/figures/part3_multiscale'])
            return
        subprocess.run([sys.executable,'scripts/summarize_part3.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'scripts/analyze_part2_failures.py','--run',str(ROOT/'runs/part3_fusion6_12_v1'),'--data-root',data,'--output',str(ROOT/'results/figures/part3_fusion_failures'),'--shot','8','--seed','0','--count','5'],cwd=ROOT,check=True)
        save('completed',required_artifacts=['results/tables/part3','docs/15_part3_results.md','results/figures/part3'])
    except Exception as e:
        save('failed',error=repr(e));raise


if __name__=='__main__':main()
