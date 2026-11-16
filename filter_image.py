"""Filter an arbitrary grayscale or RGB image without a clean reference."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from local_spectral import LocalConfig, denoise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/own_image'))
    p.add_argument('--config', type=Path, default=Path('configs/frozen.json'))
    p.add_argument('--strength', type=float, default=None)
    p.add_argument('--fixed-scale', type=int, choices=(64,128,256), default=None)
    p.add_argument('--no-conservative', action='store_true')
    args = p.parse_args()
    if args.config.exists():
        frozen = json.loads(args.config.read_text(encoding='utf-8'))
    else:
        frozen = {}
    kwargs = dict(frozen.get('local_config', {}))
    kwargs.update(fixed_scale=args.fixed_scale, conservative=not args.no_conservative,
                  strength=args.strength if args.strength is not None else frozen.get('local_strength',1.0))
    cfg = LocalConfig(**kwargs)
    # Defined luminance conversion. This course implementation restores grayscale.
    image = np.asarray(Image.open(args.input).convert('L'),dtype=np.float64)/255
    result, diag = denoise(image,cfg,return_diagnostics=True)
    args.out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(result*255).astype('uint8')).save(args.out/'restored.png')
    np.save(args.out/'restored.npy',result)
    np.savez_compressed(args.out/'diagnostics.npz',**{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
    (args.out/'decisions.json').write_text(json.dumps(dict(input=str(args.input.resolve()),
        config=asdict(cfg),tiles=diag.get('tiles',[])),ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
