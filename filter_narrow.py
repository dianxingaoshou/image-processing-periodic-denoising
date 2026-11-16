"""Explicit entry point for specialized, input-only narrow-band restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from filter_image_v3 import resolve_config
from image_io_v3 import load_gray
from narrow_denoiser import Config,config_dict,config_from_dict,denoise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/narrow_full.json'))
    parser.add_argument('--out',type=Path,default=Path('results/own_image_narrow'))
    args=parser.parse_args(argv)
    try:data,provenance=resolve_config(args.config)
    except FileNotFoundError:parser.error(f'Narrow configuration file not found: {args.config.resolve()}')
    except (ValueError,json.JSONDecodeError) as error:parser.error(str(error))
    cfg=config_from_dict(data);observed,preprocessing=load_gray(args.input,return_metadata=True)
    result,diagnostics=denoise(observed,cfg,return_diagnostics=True)
    args.out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(result*255).astype('uint8')).save(args.out/'restored.png')
    np.save(args.out/'restored.npy',result)
    np.savez_compressed(args.out/'diagnostics.npz',**{k:v for k,v in diagnostics.items() if isinstance(v,np.ndarray)})
    configuration=config_dict(cfg)
    provenance['effective_configuration_sha256']=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    record=dict(algorithm='narrow_profiles',input=str(args.input.resolve()),preprocessing=preprocessing,
        configuration=configuration,configuration_provenance=provenance,
        profiles=diagnostics['profiles'],interpretation=diagnostics['interpretation'],
        reference_metrics='not computed: no clean reference supplied')
    (args.out/'decisions.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(args.out.resolve())


if __name__=='__main__':main()
