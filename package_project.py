"""Package code and verified results; large regenerable data stay in workspace."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parent
    verification=json.loads((root/'results/verification.json').read_text(encoding='utf-8'))
    if verification.get('all_passed') is not True:
        raise RuntimeError('Complete artifact verification must pass before packaging')
    output=root/'periodic_local_code_results.zip'
    exclusions={'.venv','.git','__pycache__','.pytest_cache'}
    files=[]
    for p in root.rglob('*'):
        if not p.is_file() or p==output:continue
        rel=p.relative_to(root)
        if any(part in exclusions for part in rel.parts):continue
        posix=rel.as_posix()
        if (posix.startswith('data/clean/') or posix.startswith('data/source/images/')
                or posix.endswith('.tgz') or posix.endswith('.part')
                or posix.startswith('results/main/examples/')
                or posix.startswith('results/cli_smoke/')
                or posix.startswith('original/periodic_notch_starter/snapshot/')):
            continue
        files.append((p,posix))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,name in sorted(files,key=lambda x:x[1]):z.write(p,name)
        z.writestr('PACKAGE_CONTENTS.txt',
            'Code, frozen configuration, manifest, tests, summary metrics and figures are included.\n'
            'Large raw BSDS500 data, clean arrays and main example intermediate arrays are omitted.\n'
            'Run data_pipeline.py to download/rebuild data before filtering batch tests or verification.\n'
            'Run run_all.ps1 to regenerate the complete experiment and intermediate files.\n'
            'The existing desktop workspace retains all these omitted files.\n')
        z.writestr('PACKAGE_SHA256.json',json.dumps({name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p,name in files},indent=2))
    with zipfile.ZipFile(output) as z:
        error=z.testzip()
        if error:raise RuntimeError(f'Corrupt package member: {error}')
    print(f'{output.name}: {output.stat().st_size/1024**2:.2f} MiB, {len(files)} files, CRC passed')


if __name__=='__main__':main()
