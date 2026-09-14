import json

import pytest

from dynamic_model_loading.fault_pager_analysis import audit_pages, audit_rounds


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_page_audit_counts_real_payload_and_rejects_missing_release(tmp_path):
    size=3*1536*256*4
    key=dict(layer=0,page=1)
    load=dict(outcome='load',key=key,request='demand',bytes=size,h2d_bytes=size,
              used_bytes=size,wall_ms=1.,cuda_ms=.5)
    release=dict(outcome='release',key=key,request='eager_release',bytes=size,used_bytes=0)
    complete=[]
    for p in range(27):
        complete += [dict(load,key=dict(layer=0,page=p)),dict(release,key=dict(layer=0,page=p))]
    complete.append(dict(outcome='layer',layer=0,selected_pages=[list(range(27))],used_bytes=0))
    path=write_rows(tmp_path/'pages.jsonl',complete)
    assert audit_pages(path,128*2**20,'eager')['h2d_bytes']==size*27
    write_rows(path,[load])
    with pytest.raises(ValueError,match='unreleased'):
        audit_pages(path,128*2**20)
    write_rows(path,[dict(load,h2d_bytes=1),release])
    with pytest.raises(ValueError,match='physical'):
        audit_pages(path,128*2**20)
    write_rows(path,[complete[-1]])
    with pytest.raises(ValueError,match='reconcile'):
        audit_pages(path,128*2**20)


def test_round_audit_recomputes_commit_and_detects_draft_only_acceptance(tmp_path):
    row=dict(round=1,base=32,proposed=[3,4,5,6],target_predictions=[3,151645,5,6],
             accepted=1,emitted_accepted=1,fallback=151645,committed=[3,151645],
             target_cache=33,draft_cache=33,stop_reason='eos')
    episode=dict(ids=[3,151645],attempted=4,accepted=1)
    path=write_rows(tmp_path/'rounds.jsonl',[row])
    assert audit_rounds(path,episode)==28*36
    write_rows(path,[dict(row,committed=[3,4])])
    with pytest.raises(ValueError,match='provenance'):
        audit_rounds(path,episode)
    write_rows(path,[dict(row,target_cache=0,draft_cache=0)])
    with pytest.raises(ValueError,match='KV rollback'):
        audit_rounds(path,episode)
    write_rows(path,[row,dict(row,round=2)])
    with pytest.raises(ValueError,match='round after'):
        audit_rounds(path,episode)


@pytest.mark.parametrize('mode',['lru','prefetch'])
def test_real_cache_event_order_reconciles_evictions_and_prefetch_cancellation(tmp_path,mode):
    from dataclasses import asdict
    import torch
    from dynamic_model_loading.fault_pager import PageCache,PageKey,PagePayload
    source=PagePayload(torch.zeros(256,1536),torch.zeros(256,1536),torch.zeros(1536,256))
    cache=PageCache(128*2**20,'cpu',mode=mode)
    if mode=='prefetch':
        cache.get(PageKey(3,34),source,request='prefetch')  # Deliberately never demanded.
        cache.get(PageKey(0,0),source,request='prefetch')
    for layer in (0,1):
        for page in range(27):
            cache.get(PageKey(layer,page),source,request='demand')
        cache.record(dict(outcome='layer',layer=layer,selected_pages=[list(range(27))]))
    cache.clear()
    rows=[]
    for raw in cache.events:
        row=dict(raw)
        if 'key' in row:
            row['key']=asdict(row['key'])
        if row['outcome']=='load':
            row['evicted']=[asdict(k) for k in row['evicted']]
            row['h2d_bytes']=row['bytes']  # Synthetic CUDA byte fields; copies above are CPU-only.
        rows.append(row)
    result=audit_pages(write_rows(tmp_path/'pages.jsonl',rows),128*2**20,mode)
    assert result['evictions']>0 and result['layer']==2
    if mode=='prefetch':
        assert result['cancelled_prefetch']==1 and result['demand_hits']==1
