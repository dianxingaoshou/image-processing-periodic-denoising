"""Run v3 with explicit configuration provenance and shared float preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_io_v3 import load_gray, sha256_file


def resolve_config(path, *, use_default_config=False):
    """Return (configuration dict or None, provenance) without silent fallback."""
    path = Path(path)
    if use_default_config:
        return None, dict(source='explicit --use-default-config', path=None, sha256=None)
    if not path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {path.resolve()}. '
                                'Use --use-default-config explicitly to run uncalibrated defaults.')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    config = payload.get('configuration', payload)
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a JSON object')
    return config, dict(source='configuration file', path=str(path.resolve()), sha256=sha256_file(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/own_image_v3'))
    config_choice = p.add_mutually_exclusive_group()
    config_choice.add_argument('--config', type=Path, default=Path('configs/v3_frozen.json'))
    config_choice.add_argument('--use-default-config', action='store_true',
                   help='Explicitly use uncalibrated defaults instead of loading a frozen file')
    args = p.parse_args(argv)
    try:
        config_data, provenance = resolve_config(args.config, use_default_config=args.use_default_config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    # Resolve missing configuration before importing/running the algorithm.
    from denoiser_v3 import Config, config_from_dict, config_dict, denoise
    cfg = Config() if config_data is None else config_from_dict(config_data)
    image, preprocessing = load_gray(args.input, return_metadata=True)
    result, diag = denoise(image, cfg, return_diagnostics=True)
    result = np.asarray(result, dtype=np.float64)
    if result.shape != image.shape or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Denoiser output must match input shape and contain finite [0,1] pixels')
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(result, 0, 1) * 255).astype('uint8')).save(args.out / 'restored.png')
    np.save(args.out / 'restored.npy', result)
    np.savez_compressed(args.out / 'diagnostics.npz',
                        **{k: v for k, v in diag.items() if isinstance(v, np.ndarray)})
    serialized_config = config_dict(cfg)
    provenance['effective_configuration_sha256'] = hashlib.sha256(
        json.dumps(serialized_config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    output = dict(input=str(args.input.resolve()), input_sha256=preprocessing['input_sha256'],
                  preprocessing=preprocessing, configuration=serialized_config,
                  configuration_provenance=provenance,
                  global_assessment=diag.get('global_assessment'), tiles=diag.get('tiles', []),
                  reference_metrics='not computed: no clean reference supplied')
    (args.out / 'decisions.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
                                            encoding='utf-8')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
