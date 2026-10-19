"""Download assets on CPU, then record a real pretrained ZeroGPU invocation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from gradio_client import Client
from huggingface_hub import HfApi, get_token


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['graphcast','graphcast_small','graphcast_small_erf','genfocal_sr','genfocal_debias'])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    repo='LonghaoWang/weatherai-official'
    client=Client(repo,token=get_token(),httpx_kwargs={'timeout':120})
    client.predict(args.stage,api_name='/prepare')
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        state=json.loads(client.predict(api_name='/status'))
        if args.stage not in state['jobs']:
            client.predict(args.stage,api_name='/prepare')
            time.sleep(2)
            continue
        job=state['jobs'][args.stage]
        print(args.stage,job['status'],job.get('log',[])[-1:],flush=True)
        if job['status']=='error':
            raise RuntimeError(job['error'])
        if job['status']=='ready': break
        time.sleep(15)
    else:
        raise TimeoutError('Asset preparation exceeded 30 minutes')
    if args.prepare_only: return
    result=json.loads(client.predict(args.stage,api_name='/run_official'))
    result.update(timestamp=datetime.now(timezone.utc).isoformat(),
                  space=repo,space_revision=HfApi().space_info(repo).sha,
                  assets=job['result'],versions=state['versions'])
    output=Path(__file__).resolve().parents[1]/'docs/results'/f'{args.stage}_official_zerogpu.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='PASS': raise SystemExit(1)


if __name__=='__main__': main()
