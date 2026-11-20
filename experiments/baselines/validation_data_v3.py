"""Frozen v3 audit data and deliberately model-mismatched degradations.

Ground-truth fields in the returned masks are evaluation-only.  The first four
legacy cases delegate to the frozen v2 generator so historical recipes remain
unchanged.  New structure seeds never depend on amplitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from data_pipeline import _compact_window, sha256
from image_io_v3 import preprocess_image, structure_seed as shared_structure_seed
from run_experiments import make_sample as legacy_make_sample

SEED = 2026111603
EXTRA_KINDS = ('cubic', 'random_phase', 'corner', 'edge_strip', 'irregular')
LEGACY_KINDS = ('global', 'local', 'chirp', 'orientation')
TEXTURE_CLASSES = ('banded', 'braided', 'chequered', 'crosshatched', 'grid',
                   'grooved', 'honeycombed', 'knitted', 'lined', 'meshed',
                   'striped', 'woven')
DTD_PAGE = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html'
DTD_LABELS = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1-labels.tar.gz'
DTD_IMAGE_BASE = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/images/'


def _json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def structure_seed(image_id, kind, variant=None, base_seed=SEED):
    # Share the implementation with the v3 single-image/data boundary module;
    # variant is part of structure identity, never amplitude.
    domain=kind if variant is None else f'{kind}|variant={variant}'
    return shared_structure_seed(str(image_id),domain,seed=base_seed)


def make_sample(clean, image_id, kind, amplitude, variant=None):
    if kind in LEGACY_KINDS or kind == 'clean':
        return legacy_make_sample(clean, str(image_id), kind, amplitude)
    # The sorted audit order uses explicit corner variants in the runner; the
    # stable fallback is also independent of amplitude.
    return synthesize(clean, kind, amplitude, structure_seed(image_id, kind, variant), variant)


def synthesize(clean, kind, amplitude, seed, variant=None):
    image = np.asarray(clean, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 32 or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Expected a finite 2-D image in [0,1], dimensions >=32')
    if kind not in EXTRA_KINDS or not np.isfinite(amplitude) or not 0 < amplitude <= .5:
        raise ValueError('Invalid OOD degradation kind or amplitude')
    rng = np.random.default_rng(seed)
    h,w=image.shape; y,x=np.indices(image.shape,dtype=float)
    frequency=float(rng.uniform(.09,.145));angle=float(rng.uniform(.2,1.25))
    fx=frequency*np.cos(angle);fy=frequency*np.sin(angle)*(-1 if rng.integers(2) else 1)
    offset=float(rng.uniform(-np.pi,np.pi))
    phase=fx*x+fy*y
    true_fx=np.full_like(image,fx);true_fy=np.full_like(image,fy)
    envelope=np.ones_like(image)
    params=dict(base_frequency_cycles_per_pixel=frequency,angle_radians=angle,
                fx_cycles_per_pixel=float(fx),fy_cycles_per_pixel=float(fy),phase_offset_radians=offset)
    if kind=='cubic':
        # Third-degree phase has a quadratic local-frequency field; it is not
        # representable exactly by the estimator's quadratic phase model.
        bx=float(rng.uniform(.07,.13));by=float(rng.uniform(-.055,.055))
        cx=x-(w-1)/2;cy=y-(h-1)/2
        phase += bx*cx**3/(3*max(w-1,1)**2)+by*cy**3/(3*max(h-1,1)**2)
        true_fx += bx*cx**2/max(w-1,1)**2;true_fy += by*cy**2/max(h-1,1)**2
        envelope=.7+.3*(.5+.5*np.cos(2*np.pi*y/max(h-1,1)))
        params.update(cubic_x=bx,cubic_y=by,phase_degree=3)
    elif kind=='random_phase':
        field=ndi.gaussian_filter(rng.normal(size=image.shape),sigma=max(min(h,w)/20,1),mode='reflect')
        field-=field.mean()
        dy,dx=np.gradient(field)
        target=float(rng.uniform(.018,.033))
        factor=target/max(float(np.hypot(dy,dx).max()),1e-15)
        field*=factor;dy*=factor;dx*=factor
        phase+=field;true_fx+=dx;true_fy+=dy
        params.update(random_field='seeded Gaussian white field, Gaussian smoothed reflect',
                      smoothing_sigma_pixels=max(min(h,w)/20,1),maximum_phase_gradient=target,
                      truth_derivative='numpy.gradient of the exact generated smooth phase field')
    elif kind=='corner':
        corner=int(variant)%4 if variant is not None else int(seed)%4
        corner_names=('top_left','top_right','bottom_left','bottom_right')
        cx=(w-1)*(corner%2);cy=(h-1)*(corner//2)
        wx=float(rng.uniform(.30,.43)*w);wy=float(rng.uniform(.30,.43)*h)
        envelope=_compact_window(x,cx,wx)*_compact_window(y,cy,wy)
        params.update(corner=corner_names[corner],corner_index=corner,center_xy_pixels=[float(cx),float(cy)],
                      halfwidth_xy_pixels=[wx,wy])
    elif kind=='edge_strip':
        side=int(variant)%4 if variant is not None else int(seed)%4
        side_names=('left','right','top','bottom')
        if side<2:
            center=(.025 if side==0 else .975)*(w-1);halfwidth=float(rng.uniform(.105,.16)*w)
            envelope=_compact_window(x,center,halfwidth)*(.75+.25*np.cos(np.pi*y/max(h-1,1))**2)
        else:
            center=(.025 if side==2 else .975)*(h-1);halfwidth=float(rng.uniform(.105,.16)*h)
            envelope=_compact_window(y,center,halfwidth)*(.75+.25*np.cos(np.pi*x/max(w-1,1))**2)
        params.update(edge=side_names[side],edge_index=side,strip_center_pixels=float(center),
                      strip_halfwidth_pixels=halfwidth)
    elif kind=='irregular':
        cx=float(rng.uniform(.14,.86));cy=float(rng.uniform(.14,.86))
        ax=float(rng.uniform(.22,.36));ay=float(rng.uniform(.22,.36));warp=float(rng.uniform(-np.pi,np.pi))
        u=x/max(w-1,1)-cx;v=y/max(h-1,1)-cy
        radial=np.sqrt((u/ax)**2+(v/ay)**2)
        shape=np.maximum(radial+.14*np.sin(5*np.arctan2(v,u)+warp)+.05*np.sin(13*u+9*v),0)
        envelope=np.where(shape<1,.5*(1+np.cos(np.pi*np.minimum(shape,1))),0.)
        params.update(center_xy_fraction=[cx,cy],radii_xy_fraction=[ax,ay],radial_warp_phase=warp,
                      mask='compact warped elliptical raised cosine')
    envelope[envelope<1e-6]=0.
    if envelope.max()>0:envelope/=envelope.max()
    norms=np.hypot(true_fy,true_fx)
    if norms.min()<.025 or norms.max()>.45:
        raise RuntimeError('Generated phase frequency left the declared [0.025,0.45] cycles/pixel interval')
    added=amplitude*envelope*np.sin(2*np.pi*phase+offset)
    raw=image+added;noisy=np.clip(raw,0,1)
    contaminated=envelope>.01;clean_mask=envelope==0;transition=~(contaminated|clean_mask)
    meta=dict(kind=kind,amplitude=float(amplitude),seed=int(seed),variant=variant,shape=[h,w],
        parameters=params,frequency_unit='cycles/pixel',frequency_min=float(norms.min()),frequency_max=float(norms.max()),
        clipping_fraction=float(np.mean((raw<0)|(raw>1))),contaminated_fraction=float(contaminated.mean()),
        clean_fraction=float(clean_mask.mean()),transition_fraction=float(transition.mean()),
        preclip_noise_rms=float(np.sqrt(np.mean(added**2))),postclip_noise_rms=float(np.sqrt(np.mean((noisy-image)**2))),
        model_mismatch=kind in ('cubic','random_phase'),
        mask_definition='contaminated envelope>0.01; clean envelope==0; transition remainder; truth evaluation only')
    masks=dict(contaminated=contaminated,clean=clean_mask,transition=transition,envelope=envelope,
               true_noise=noisy-image,preclip_noise=added,true_fy=true_fy,true_fx=true_fx)
    return noisy,meta,masks


def _download(url,target):
    import requests
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():return target
    temporary=target.with_suffix(target.suffix+'.part')
    with requests.get(url,stream=True,timeout=(20,90)) as response:
        response.raise_for_status()
        with temporary.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk:f.write(chunk)
    temporary.replace(target);return target


def _save_reference(source,target,size=256):
    clean,_=preprocess_image(source,size=size)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.save(target,clean)
    Image.fromarray(np.rint(clean*255).astype('uint8')).save(target.with_suffix('.png'))


def build_manifest(root='data'):
    root=Path(root);target=root/'v3_manifest.json'
    old_path=root/'manifest.json';v2_path=root/'optimization_manifest.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));v2=json.loads(v2_path.read_text(encoding='utf-8'))
    used={r['id'] for r in old['images'] if r['split']=='test'}|{r['id'] for r in v2['images'] if r['split']=='audit_test'}
    pool=sorted(p for p in (root/'source/images/test').glob('*.jpg') if p.stem not in used)
    if len(pool)!=80:raise ValueError(f'Expected 80 unused official test images, got {len(pool)}')
    chosen=sorted(np.random.default_rng(SEED).choice(pool,60,replace=False))
    selection=dict(seed=SEED,available_official_test_pool=80,n_audit_v3=60,
                   audit_v3_ids=[p.stem for p in chosen],selection_uses_image_content=False,
                   selected_before_preprocessing=True)
    protocol=dict(schema_version=3,selection=selection,
        primary=dict(split='audit_v3',n_images=60,kinds=LEGACY_KINDS,amplitudes=[.06,.12,.18],clean_controls=True,
                     n_samples=780,methods=['input','v2_frozen','v3_coupled_P0','v3_decoupled','v3_unprotected'],expected_rows=3900),
        model_mismatch_and_location=dict(split='audit_v3',image_ids=selection['audit_v3_ids'][:20],n_images=20,
                     kinds=EXTRA_KINDS,amplitudes=[.12],n_samples=100,n_methods=5,expected_rows=500,
                     corner_placement='audit subset index modulo4, covering every corner five times'),
        textures=dict(dataset='DTD official test1 fixed subset',n_images=12,kinds=['clean','global','chirp'],amplitudes=[0,.12,.12],
                     use_for_parameter_selection=False,n_samples=36,
                     caveat='Original photographs have no newly added interference; not guaranteed physically noise-free.'),
        real_examples=dict(n_cases=2,configuration='one frozen configuration, no image-specific tuning',reference_metrics=False),
        calibration=dict(split='dev',ids=[r['id'] for r in old['images'] if r['split']=='dev'],
                     note='Only original development images and already-known analytic controls may choose parameters; DTD/audit are not tuning data.'),
        frequency_estimation=dict(unit='cycles/pixel',comparison='at synthesis tile center, min(norm(fhat-f),norm(fhat+f))',
                     denominator='all contaminated tile centers; separately report candidate coverage and actual processing coverage',
                     map_warning='overlap-averaged center-frequency display is not a full quadratic phase frequency field'),
        attribution='Keep frozen v2 unchanged; compare coupled-P0, decoupled and unprotected using the same new estimator residuals; P2 optional and separately switched.')
    # Freeze filename selection and exact experiment intent before touching new
    # audit pixels. Existing v2 files are read only.
    _json('v3_protocol.json',protocol)
    records=[]
    for r in old['images']:
        if r['split']=='dev':records.append(dict(r,split='dev'))
        else:records.append(dict(r,split='regression_v1'))
    for r in v2['images']:
        if r['split']=='audit_test':records.append(dict(r,split='regression_v2'))
    for p in chosen:
        destination=root/'clean/audit_v3'/f'{p.stem}.npy'
        _save_reference(p,destination)
        records.append(dict(id=p.stem,image_id=p.stem,split='audit_v3',official_split='test',
            source_jpg=p.relative_to(root).as_posix(),source_sha256=sha256(p),npy_path=destination.relative_to(root).as_posix(),
            clean_npy=destination.relative_to(root).as_posix(),clean_png=destination.with_suffix('.png').relative_to(root).as_posix(),
            clean_sha256=sha256(destination)))
    if len(records)!=210 or len({r['id'] for r in records})!=210:raise ValueError('Invalid v3 split inventory')
    manifest=dict(schema_version=3,dataset='BSDS500 images plus separate DTD texture stress subset and unpaired real cases',
        source_manifest_sha256=sha256(old_path),v2_manifest_sha256=sha256(v2_path),selection=selection,
        protocol_path='../v3_protocol.json',protocol_sha256=sha256(Path('v3_protocol.json')),
        preprocessing='RGB short-side Lanczos resize; centered square crop; float 0.2125R+0.7154G+0.0721B; no contrast rescaling',
        images=records,textures=[],real_examples=[])
    _json(target,manifest);return manifest


def build_textures(root='data'):
    root=Path(root);folder=root/'textures_v3'
    archive=_download(DTD_LABELS,folder/'dtd-r1.0.1-labels.tar.gz')
    with tarfile.open(archive,'r:gz') as z:
        candidates=[]
        for member in z:
            parts=PurePosixPath(member.name)
            if parts.is_absolute() or '..' in parts.parts:raise ValueError('Unsafe DTD archive member')
            if member.isfile() and parts.name=='test1.txt':
                candidates=z.extractfile(member).read().decode('utf-8').splitlines();break
    if not candidates:raise ValueError('Official DTD test1 list missing')
    chosen=[];rng=np.random.default_rng(SEED)
    for category in TEXTURE_CLASSES:
        pool=sorted(s for s in candidates if s.startswith(category+'/'))
        if not pool:raise ValueError('Missing DTD category '+category)
        chosen.append(str(rng.choice(pool)))
    _json(folder/'selection.json',dict(seed=SEED,classes=TEXTURE_CLASSES,official_split='test1',selected=chosen,
                                     selection_before_download=True,source=DTD_PAGE))
    records=[]
    for name in chosen:
        category,filename=name.split('/')
        url=DTD_IMAGE_BASE+quote(name,safe='/')
        source=_download(url,folder/'source'/category/filename)
        # Decode before accepting a successful HTTP response as an image.
        with Image.open(source) as image:image.verify()
        ident='dtd_'+Path(filename).stem;target=folder/'clean'/f'{ident}.npy'
        _save_reference(source,target)
        records.append(dict(id=ident,image_id=ident,split='texture_test',dataset='DTD',category=category,official_split='test1',
            source_url=url,source_jpg=source.relative_to(root).as_posix(),source_sha256=sha256(source),
            npy_path=target.relative_to(root).as_posix(),clean_sha256=sha256(target),
            note='Natural texture photo with no newly added interference; not a guaranteed physically noise-free reference.'))
    provenance=dict(dataset='Describable Textures Dataset',official_page=DTD_PAGE,labels_url=DTD_LABELS,
        labels_sha256=sha256(archive),image_download='12 direct official JPEGs, not the625MB full archive',
        use_terms='Official page makes the data available for research purposes; no broader permissive image license asserted.',
        citation='M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi. Describing Textures in the Wild. CVPR2014.',
        selection='one seeded filename per12 preregistered categories from official test1; no image-content selection',records=records)
    _json(folder/'provenance.json',provenance)
    return records


def build_real_examples():
    folder=Path('third_party/real_examples')
    entries=[dict(id='opencv_period_input',filename='opencv_period_input.jpg',
        url='https://raw.githubusercontent.com/opencv/opencv/4.x/doc/tutorials/imgproc/periodic_noise_removing_filter/images/period_input.jpg',
        source_page='https://docs.opencv.org/4.13.0/d2/d0b/tutorial_periodic_noise_removing_filter.html',
        interpretation='Official OpenCV tutorial explicitly describes this as a real-world periodic-noise image.',
        use_terms='Official OpenCV repository sample; repository Apache2.0 terms do not establish ownership of every photographed subject.'),
        dict(id='arpenos_canvas_carry_on_002',filename='arpenos_canvas_carry_on_002.jpg',
        url='https://members.loria.fr/FSur/software/ACARPENOS/Rockwell%20-%20Carry%20On%20-%20002.jpg',
        source_page='https://members.loria.fr/FSur/software/ARPENOS/',
        interpretation='Author-provided painting canvas-pattern removal case; texture may be physical substrate/content, not paired sensor-noise truth.',
        use_terms='ARPENOS author page permits academic research use; original painting image rights retained by their owners.')]
    for entry in entries:
        path=_download(entry['url'],folder/entry['filename'])
        with Image.open(path) as image:
            entry['shape']=[image.height,image.width];image.verify()
        entry.update(path=path.as_posix(),sha256=sha256(path),reference_available=False,
                     evaluation='qualitative fixed-configuration outputs only; no PSNR or SSIM',
                     processing_recommendation='shared float grayscale; optionally fix longest side512 for every real case, no per-image tuning')
    _json(folder/'provenance.json',dict(records=entries,matlab_or_octave_available=False,
        baseline_equivalence='No MATLAB/Octave runtime found. Original-versus-port numerical comparison is not claimed completed.',
        original_example_warning='mandril_gray.tif and boat.png are clean example references with synthesized noise in ARPENOS_example.m; they are not labeled real-noisy cases.'))
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data');p.add_argument('--skip-downloads',action='store_true')
    args=p.parse_args()
    manifest_path=Path(args.root)/'v3_manifest.json'
    previous=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    manifest=build_manifest(args.root)
    if not args.skip_downloads:
        manifest['textures']=build_textures(args.root);manifest['real_examples']=build_real_examples()
    else:
        # Skipping downloads must not erase previously frozen texture/real IDs.
        manifest['textures']=previous.get('textures',[])
        manifest['real_examples']=previous.get('real_examples',[])
    _json(manifest_path,manifest)
    print(f'v3 data: {len(manifest["images"])} BSDS references, {len(manifest["textures"])} DTD textures, {len(manifest["real_examples"])} unpaired real cases')


if __name__=='__main__':main()
