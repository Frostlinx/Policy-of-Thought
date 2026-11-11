#!/usr/bin/env python3
"""Download pinned Qwen weights via hf-mirror. Reads optional HF_TOKEN only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

from download_lcb import _MirrorRedirects, file_sha256
from mirror_transfer import aria2_download

REPO = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/Qwen3-4B-Instruct-2507')
    parser.add_argument('--no-proxy', action='store_true')
    parser.add_argument('--aria2', action='store_true', help='Use 8 resumed connections with upstream integrity checks')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    handlers = [_MirrorRedirects()]
    if args.no_proxy:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)

    def fetch(name):
        if '/' in name or name.startswith('.'):
            raise ValueError('invalid model file')
        target = root/name
        partial = root/(name+'.part')
        metadata_path = root/(name+'.download.json')
        if target.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            if meta['revision'] == REVISION and meta['sha256'] == file_sha256(target):
                print('Verified existing '+name,flush=True)
                return meta
        url = f'https://hf-mirror.com/{REPO}/resolve/{REVISION}/{name}?download=true'
        if args.aria2:
            aria2_download(url,partial,no_proxy=args.no_proxy)
            digest=file_sha256(partial)
            partial.replace(target)
            meta={'filename':name,'sha256':digest,'bytes':target.stat().st_size,
                  'revision':REVISION,'source':url}
            metadata_path.write_text(json.dumps(meta)+'\n')
            print('Downloaded '+name+' '+str(meta['bytes'])+' bytes',flush=True)
            return meta
        for attempt in range(5):
            headers = {'User-Agent':'policy-of-thoughts/0.5.1'}
            token = os.getenv('HF_TOKEN')
            if token:
                headers['Authorization'] = 'Bearer '+token
            offset = partial.stat().st_size if partial.exists() else 0
            if offset:
                headers['Range'] = f'bytes={offset}-'
            try:
                request = urllib.request.Request(url,headers=headers)
                with opener.open(request,timeout=180) as response:
                    append = response.status == 206 and offset > 0
                    if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('range mismatch')
                    expected = int(response.headers.get('Content-Length',0))
                    received = 0
                    with partial.open('ab' if append else 'wb') as out:
                        while True:
                            chunk = response.read(4*1024*1024)
                            if not chunk: break
                            out.write(chunk); received += len(chunk)
                    if expected and received != expected:
                        raise RuntimeError('incomplete download')
                digest=file_sha256(partial)
                partial.replace(target)
                meta={'filename':name,'sha256':digest,'bytes':target.stat().st_size,
                      'revision':REVISION,'source':url}
                metadata_path.write_text(json.dumps(meta)+'\n')
                print('Downloaded '+name+' '+str(meta['bytes'])+' bytes',flush=True)
                return meta
            except Exception as exc:
                print('Retry '+name+' '+str(attempt+1)+' ('+type(exc).__name__+')',flush=True)
                if attempt==4: raise RuntimeError('Model file download failed: '+name) from None
                time.sleep(min(20,2**attempt))

    names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
           'vocab.json','merges.txt','model.safetensors.index.json']
    records=[fetch(name) for name in names]
    index=json.loads((root/'model.safetensors.index.json').read_text())
    shards=sorted(set(index['weight_map'].values()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records.extend(pool.map(fetch,shards))
    # Parse safetensors headers and verify each shard's declared ranges before use.
    import struct
    for name in shards:
        with (root/name).open('rb') as f:
            header_size=struct.unpack('<Q',f.read(8))[0]
            if header_size > 100_000_000: raise RuntimeError('invalid safetensors header')
            header=json.loads(f.read(header_size))
        max_end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        if 8+header_size+max_end != (root/name).stat().st_size:
            raise RuntimeError('safetensors shard size mismatch')
    (root/'pot_download_manifest.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,
        'endpoint':'https://hf-mirror.com','files':records},indent=2)+'\n')
    print('MODEL_READY '+str(root),flush=True)

if __name__=='__main__':
    main()
