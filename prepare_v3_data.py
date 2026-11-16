"""Restore fixed v3 inputs without reselection or replacing frozen manifests."""
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from data_pipeline import _download,_extract_images,sha256
from image_io_v3 import preprocess_image


def fetch(url,path,expected):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with path.open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    if chunk:f.write(chunk)
    if sha256(path)!=expected:raise ValueError('Source hash mismatch: '+str(path))


def main():
    data=Path('data');m=json.loads((data/'v3_manifest.json').read_text(encoding='utf-8'))
    if not all((data/r['source_jpg']).exists() for r in m['images']):
        archive=data/'source/BSR_bsds500.tgz'
        if not archive.exists():_download(archive)
        original=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        if sha256(archive)!=original['archive_sha256']:raise ValueError('Official BSDS500 archive hash mismatch')
        _extract_images(archive,data/'source/images')
    for r in m['textures']:fetch(r['source_url'],data/r['source_jpg'],r['source_sha256'])
    for r in m['real_examples']:fetch(r['url'],Path(r['path']),r['sha256'])
    for r in m['images']+m['textures']:
        source=data/r['source_jpg'];target=data/r['npy_path']
        if sha256(source)!=r['source_sha256']:raise ValueError('Source changed: '+r['id'])
        if not target.exists():
            clean,_=preprocess_image(source,256);target.parent.mkdir(parents=True,exist_ok=True)
            np.save(target,clean);Image.fromarray(np.rint(clean*255).astype('uint8')).save(target.with_suffix('.png'))
        if sha256(target)!=r['clean_sha256']:raise ValueError('Reference changed: '+r['id'])
    print('All fixed210 BSDS,12 DTD and2 unpaired real input hashes passed')


if __name__=='__main__':main()
