import json
import sys
import validation_data_v3 as data


def test_skip_downloads_keeps_frozen_external_records(tmp_path,monkeypatch):
    path=tmp_path/'v3_manifest.json'
    previous=dict(textures=[dict(id='fixed_texture')],real_examples=[dict(id='fixed_real')])
    path.write_text(json.dumps(previous),encoding='utf-8')
    monkeypatch.setattr(data,'build_manifest',lambda root:dict(images=[],textures=[],real_examples=[]))
    monkeypatch.setattr(sys,'argv',['validation_data_v3.py','--root',str(tmp_path),'--skip-downloads'])
    data.main()
    after=json.loads(path.read_text(encoding='utf-8'))
    assert after['textures']==previous['textures']
    assert after['real_examples']==previous['real_examples']
