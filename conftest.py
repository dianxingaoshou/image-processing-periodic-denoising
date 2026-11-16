"""Make absent external data explicit without hiding full-validation failures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--require-bsds', action='store_true', default=False,
                     help='Fail BSDS integration tests when any required reference is absent.')


def _missing_bsds_inputs(root, dataset):
    """Return missing input paths; invalid manifests remain errors, not skips."""
    root=Path(root)
    if dataset=='legacy':
        manifest=root/'data/manifest.json'
        key='clean_npy'
    elif dataset=='v3':
        manifest=root/'data/v3_manifest.json'
        key='npy_path'
    else:
        raise ValueError(f'Unknown BSDS integration dataset: {dataset}')
    if not manifest.is_file():return [manifest]
    payload=json.loads(manifest.read_text(encoding='utf-8-sig'))
    rows=payload['images']
    if dataset=='v3':rows=[row for row in rows if row['split']=='audit_v3']
    paths=[root/'data'/row[key] for row in rows]
    return [path for path in paths if not path.is_file()]


@pytest.fixture(autouse=True)
def require_prepared_bsds(request):
    marker=request.node.get_closest_marker('requires_bsds')
    if marker is None:return
    dataset=marker.kwargs.get('dataset','legacy')
    missing=_missing_bsds_inputs(request.config.rootpath,dataset)
    if not missing:return
    reason=(f'BSDS integration not verified: {len(missing)} required input(s) absent '
            f'(first: {missing[0]}). Run python prepare_v3_data.py, then pytest --require-bsds.')
    if request.config.getoption('--require-bsds'):
        pytest.fail(reason,pytrace=False)
    pytest.skip(reason)
