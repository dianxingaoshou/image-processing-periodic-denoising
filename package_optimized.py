"""Bundle both versions and verified v2 evidence, omitting regenerable bulk data."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parent
    verification=json.loads((root/'results/optimization_v2/verification.json').read_text(encoding='utf-8'))
    if not verification['all_passed']:raise RuntimeError('V2 verification must pass first')
    output=root/'periodic_local_optimized_v2.zip';files=[]
    for path in root.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(root);name=rel.as_posix()
        if any(part in {'.venv','.git','__pycache__','.pytest_cache'} for part in rel.parts):continue
        if path.suffix in {'.zip','.tgz','.part','.pyc'}:continue
        if (name.startswith('data/clean/') or name.startswith('data/source/images/')
                or name.startswith('original/periodic_notch_starter/snapshot/')
                or name.startswith('results/main/examples/') or name.startswith('results/cli_smoke/')
                or name.startswith('results/own_image_v2/')):continue
        if name.startswith('results/optimization_v2/audit/examples/') and path.suffix in {'.npy','.npz'}:continue
        files.append((path,name))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in sorted(files,key=lambda x:x[1]):z.write(path,name)
        z.writestr('PACKAGE_CONTENTS.txt',
            'Contains v1 and v2 source, frozen configurations, protocols, tests and verified numeric/visual results.\n'
            'Large raw BSDS500 data, reference arrays and example intermediate arrays are omitted.\n'
            'Use prepare_optimization_data.py to rebuild all150 references from the official source.\n'
            'Use run_optimization.ps1 for a full rerun. The original desktop workspace retains all arrays.\n')
        z.writestr('PACKAGE_SHA256.json',json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in files},indent=2))
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Corrupt zip member: '+bad)
    print(f'{output.name}: {output.stat().st_size/1024**2:.2f} MiB, {len(files)} files, CRC passed')


if __name__=='__main__':main()
