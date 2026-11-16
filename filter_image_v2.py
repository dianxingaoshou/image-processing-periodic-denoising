"""Run the optimized local denoiser on a user image, without reference metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from optimized_denoiser import V2Config,config_from_dict,config_dict,denoise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,default=Path('results/own_image_v2'))
    p.add_argument('--config',type=Path,default=Path('configs/optimized_v2.json'))
    args=p.parse_args()
    cfg=config_from_dict(json.loads(args.config.read_text(encoding='utf-8'))['configuration']) if args.config.exists() else V2Config()
    image=np.asarray(Image.open(args.input).convert('L'),dtype=float)/255
    result,diag=denoise(image,cfg,return_diagnostics=True)
    args.out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(result*255).astype('uint8')).save(args.out/'restored.png')
    np.save(args.out/'restored.npy',result)
    np.savez_compressed(args.out/'diagnostics.npz',**{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
    (args.out/'decisions.json').write_text(json.dumps(dict(config=config_dict(cfg),input=str(args.input.resolve()),
        global_assessment=diag['global_assessment'],tiles=diag['tiles']),indent=2,ensure_ascii=False),encoding='utf-8')
    print(args.out.resolve())


if __name__=='__main__':main()
