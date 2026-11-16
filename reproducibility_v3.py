"""Source, binary dependency, environment and ordered-audit freeze checks.

Snapshots use paths relative to a caller-selected base directory where possible.
Moving a project does not itself invalidate a snapshot, while changing source,
numeric dependencies or the audit ID order does. No reference pixels are read.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

DEFAULT_DEPENDENCIES = ('numpy', 'scipy', 'scikit-image', 'Pillow', 'matplotlib', 'pytest', 'requests')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _runtime():
    return dict(implementation=platform.python_implementation(),
                python_version=platform.python_version(), python_build=list(platform.python_build()),
                architecture=platform.architecture()[0], machine=platform.machine(),
                byteorder=sys.byteorder, platform_system=platform.system())


def _dependency_state(names, hash_artifacts):
    state = {}
    for name in sorted(set(names), key=str.casefold):
        dist = metadata.distribution(name)
        item = dict(version=dist.version)
        if hash_artifacts:
            artifact_hashes = []
            for relative in sorted(dist.files or (), key=str):
                label = str(relative).replace('\\', '/')
                if label.endswith('.pyc') or '__pycache__/' in label:
                    continue
                path = Path(dist.locate_file(relative))
                if path.is_file():
                    artifact_hashes.append((label, sha256_file(path)))
                else:
                    artifact_hashes.append((label, 'MISSING'))
            if not artifact_hashes:
                raise RuntimeError(f'Cannot freeze installed artifacts for dependency {name}')
            item.update(artifact_digest=_canonical_digest(artifact_hashes),
                        artifact_count=len(artifact_hashes),
                        artifact_hash_policy='SHA256 of installed distribution files; excludes pyc/cache')
        state[name] = item
    return state


def _ids(audit_ids):
    if audit_ids is None:
        raise ValueError('Actual audit_ids must be supplied so their exact order is checked')
    ids = [str(x) for x in audit_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('audit_ids must be a nonempty ordered sequence without duplicates')
    return ids


def _file_map(paths, base_dir):
    output = {}
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Frozen dependency file missing: {path}')
        try:
            label = path.relative_to(base_dir).as_posix()
        except ValueError:
            label = path.as_posix()
        if label in output:
            raise ValueError(f'Duplicate frozen source path: {label}')
        output[label] = sha256_file(path)
    if not output:
        raise ValueError('At least one source/protocol/manifest file must be frozen')
    return dict(sorted(output.items()))


def capture_snapshot(paths, audit_ids, dependency_names=DEFAULT_DEPENDENCIES, *,
                     metadata=None, base_dir=None, hash_dependencies=True):
    """Freeze explicit transitive source/data/metric/baseline paths and runtime.

    Callers should include data generators, metrics, baseline implementation,
    imported helpers, all v3 algorithms, protocol and manifest in ``paths``.
    Third-party compiled numeric artifacts are hashed by default, not just their
    version strings. The snapshot is JSON serializable and carries its own seal.
    """
    base = Path(base_dir or Path.cwd()).resolve()
    dependency_names = tuple(dependency_names)
    snapshot = dict(schema_version=1, files=_file_map(paths, base), audit_ids=_ids(audit_ids),
                    dependency_names=sorted(set(dependency_names), key=str.casefold),
                    hash_dependencies=bool(hash_dependencies),
                    dependencies=_dependency_state(dependency_names, hash_dependencies),
                    runtime=_runtime(), metadata=metadata or {})
    snapshot['snapshot_sha256'] = _canonical_digest(snapshot)
    return snapshot


def verify_snapshot(snapshot, paths=None, audit_ids=None, dependency_names=None, *,
                    base_dir=None, raise_on_error=True, hash_dependencies=None):
    """Check the frozen inventory and exact audit ID sequence before evaluation.

    Returns a report. With default ``raise_on_error=True`` any changed source,
    runtime, dependency artifact, audit order or corrupted snapshot raises.
    Supplying a reduced paths list is rejected instead of silently narrowing the
    freeze. Absolute executable/workspace locations are not identity checks.
    """
    base = Path(base_dir or Path.cwd()).resolve()
    checks = {}; errors = []

    def check(name, fn):
        try:
            ok = bool(fn())
            checks[name] = ok
            if not ok:
                errors.append(name)
        except Exception as exc:
            checks[name] = False
            errors.append(f'{name}: {type(exc).__name__}: {exc}')

    seal = {k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}
    check('snapshot_integrity', lambda: snapshot.get('schema_version') == 1 and
          snapshot.get('snapshot_sha256') == _canonical_digest(seal))
    def ordered_ids_match():
        actual = _ids(audit_ids)
        expected = snapshot['audit_ids']
        if actual != expected:
            first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
            raise ValueError(f'Audit ID order/count differs at position {first}; actual count={len(actual)}, frozen count={len(expected)}')
        return True
    check('audit_ids_exact_order', ordered_ids_match)
    actual_paths = list(snapshot.get('files', {})) if paths is None else paths
    def source_inventory_matches():
        actual = _file_map(actual_paths, base)
        expected = snapshot['files']
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
            raise ValueError(f'Changed files={changed}; missing files={missing}; extra files={extra}')
        return True
    check('source_manifest_protocol_inventory', source_inventory_matches)
    names = snapshot.get('dependency_names', []) if dependency_names is None else list(dependency_names)
    check('dependency_inventory', lambda: sorted(set(names), key=str.casefold) == snapshot['dependency_names'])
    policy = snapshot.get('hash_dependencies', True) if hash_dependencies is None else bool(hash_dependencies)
    check('dependency_hash_policy', lambda: policy == snapshot.get('hash_dependencies', True))
    def dependency_artifacts_match():
        actual = _dependency_state(names, policy)
        expected = snapshot['dependencies']
        if actual != expected:
            changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
            raise ValueError(f'Changed numerical dependency distributions={changed}')
        return True
    check('numeric_dependency_versions_and_artifacts', dependency_artifacts_match)
    check('numeric_runtime', lambda: _runtime() == snapshot['runtime'])
    report = dict(all_passed=all(checks.values()), checks=checks, errors=errors,
                  snapshot_sha256=snapshot.get('snapshot_sha256'))
    if not report['all_passed'] and raise_on_error:
        raise RuntimeError('Frozen experiment mismatch: ' + '; '.join(errors))
    return report


__all__ = ['DEFAULT_DEPENDENCIES', 'capture_snapshot', 'verify_snapshot', 'sha256_file']
