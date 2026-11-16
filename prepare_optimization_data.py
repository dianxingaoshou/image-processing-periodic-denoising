"""Rebuild the preregistered audit references from the official BSDS500 images."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline import build_dataset,_extract_images,_preprocess,sha256


def main():
    root=Path('data');manifest=json.loads((root/'optimization_manifest.json').read_text(encoding='utf-8'))
    build_dataset(root)
    if not all((root/r['source_jpg']).exists() for r in manifest['images']):
        _extract_images(root/'source/BSR_bsds500.tgz',root/'source/images')
    for record in manifest['images']:
        source=root/record['source_jpg'];target=root/record['npy_path']
        if sha256(source)!=record['source_sha256']:raise ValueError('Source hash mismatch: '+str(source))
        if not target.exists():
            clean=_preprocess(source,256);target.parent.mkdir(parents=True,exist_ok=True);np.save(target,clean)
            Image.fromarray(np.rint(clean*255).astype('uint8')).save(target.with_suffix('.png'))
        if sha256(target)!=record['clean_sha256']:raise ValueError('Reference hash mismatch: '+str(target))
    print('All 150 references match the frozen optimization manifest')


if __name__=='__main__':main()
