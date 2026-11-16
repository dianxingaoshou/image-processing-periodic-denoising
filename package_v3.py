"""Package frozen v3 code and evidence, retaining historical source baselines."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parent
    report=json.loads((root/'results/v3/verification.json').read_text(encoding='utf-8'))
    if not report['all_passed']:raise RuntimeError('V3 verification must pass before packaging')
    output=root/'periodic_phase_conservative_v3.zip';files=[]
    for path in root.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(root);name=rel.as_posix()
        if any(part in {'.venv','.git','__pycache__','.pytest_cache'} for part in rel.parts):continue
        if path.suffix in {'.zip','.tgz','.part','.pyc'}:continue
        if name.startswith('data/source/images/') or name.startswith('data/clean/'):continue
        if name.startswith('original/periodic_notch_starter/snapshot/'):continue
        if name.startswith(('results/main/examples/','results/main/figures/','results/main/failure_audit/',
            'results/optimization_v2/audit/examples/','results/optimization_v2/supplement/texture_arrays',
            'results/cli_smoke/','results/own_image_')):continue
        if name.startswith('results/v3/') and '/examples/' in name and path.suffix in {'.npy','.npz'}:continue
        files.append((path,name))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in sorted(files,key=lambda item:item[1]):z.write(path,name)
        z.writestr('PACKAGE_CONTENTS.txt',
            'Contains frozen v3 source and configuration, P0/P1 tests and evidence, historical v1/v2 source and numeric context.\n'
            'BSDS500 raw archive, raw JPEGs and reference arrays, plus bulky example intermediate arrays are omitted.\n'
            'prepare_v3_data.py restores all fixed inputs from the original URLs and verifies hashes.\n'
            'run_v3.ps1 reruns the primary experiment. Historical full-image galleries remain in the original workspace/bundles.\n'
            'Numeric dependency binary hashes are strict; matching runtime is needed for exact frozen verification.\n')
        z.writestr('PACKAGE_SHA256.json',json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in files},indent=2))
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Corrupt archive member: '+bad)
    print(f'{output.name}: {output.stat().st_size/1024**2:.2f} MiB, {len(files)} files, CRC passed')


if __name__=='__main__':main()
