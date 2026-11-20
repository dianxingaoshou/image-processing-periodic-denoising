"""Restore one image with the integrated periodic interference pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import import_module
import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_io_v3 import load_gray, sha256_file
from project_pipeline import DEFAULT_GENERAL_CONFIG, DEFAULT_NARROW_CONFIG, load_config


# Keep historical switches usable from experiment scripts and archived commands.
VERSIONS = {
    "collision": ("project_collision", "preceding protected-band reconstruction"),
    "peak": ("project_peak", "distant texture peak correction"),
    "guarded": ("project_guarded", "clipping-supported carrier"),
    "weak": ("project_weak", "weak-carrier recovery"),
    "stripes": ("project_stripes", "narrow harmonic review"),
    "weave": ("project_weave", "broad lattice protection"),
    "previous": ("project_refined", "early texture protection"),
    "baseline": ("project_pipeline", "first integrated pipeline"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input image. RGB/RGBA is converted to luminance.')
    versions = parser.add_mutually_exclusive_group()
    for switch, (_, description) in VERSIONS.items():
        versions.add_argument('--'+switch, dest='version', action='store_const', const=switch, help=description)
    parser.add_argument('--out', type=Path, help='Output directory (default under results, separate for each version)')
    args = parser.parse_args(argv)
    algorithm = VERSIONS[args.version][0] if args.version else 'project_phase'
    module = import_module(algorithm)
    suffix = '_'+args.version if args.version else ''
    if args.out is None:
        args.out = Path(f'results/own_image_project{suffix}')
    output_names = ('restored.png', 'restored.npy', 'diagnostics.npz', 'decisions.json')
    try:
        observed, preprocessing = load_gray(args.input, return_metadata=True)
        for name in output_names:
            target = args.out / name
            if target.exists() and target.samefile(args.input):
                parser.error(f'Output file would overwrite the input image: {target}')
        config = load_config()
        result, diagnostics = module.denoise(observed, config, return_diagnostics=True)
        result = np.asarray(result, dtype=np.float64)
        if (result.shape != observed.shape or not np.isfinite(result).all()
                or np.any((result < 0) | (result > 1))):
            raise ValueError('Denoiser output must match input shape and contain finite [0,1] pixels')
        configuration = asdict(config)
        if hasattr(module, 'routing_parameters'):
            configuration['routing'] = module.routing_parameters()
        effective_bytes = json.dumps(configuration, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')
        provenance = {
            'general': {'file': DEFAULT_GENERAL_CONFIG.name, 'sha256': sha256_file(DEFAULT_GENERAL_CONFIG)},
            'narrow': {'file': DEFAULT_NARROW_CONFIG.name, 'sha256': sha256_file(DEFAULT_NARROW_CONFIG)},
            'effective_configuration_sha256': hashlib.sha256(effective_bytes).hexdigest(),
        }
        maps = {name: value for name, value in diagnostics.items() if isinstance(value, np.ndarray)}
        record = {name: value for name, value in diagnostics.items() if name not in maps}
        record.update(algorithm=algorithm,
                      input=str(args.input.resolve()),
                      preprocessing=preprocessing, configuration=configuration,
                      configuration_provenance=provenance,
                      reference_metrics='not computed: no clean reference supplied')
        decisions_json = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
        args.out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(result * 255).astype('uint8')).save(args.out / 'restored.png')
        np.save(args.out / 'restored.npy', result)
        np.savez_compressed(args.out / 'diagnostics.npz', **maps)
        (args.out / 'decisions.json').write_text(decisions_json, encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{diagnostics['route']} ({diagnostics['reason']})  {args.out.resolve()}")


if __name__ == '__main__':
    main()
