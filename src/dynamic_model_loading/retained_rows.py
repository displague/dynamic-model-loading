"""Observed-zero packets with optional bounded per-layer LRU retention.

The original dense outgoing arithmetic is retained. Cache misses are observed
residency misses, never predictions that a contribution is unimportant.
"""
from collections import OrderedDict
import time
import numpy as np
import torch


def retention_plan(state, active, capacity):
    """Return hits/misses/inserts and mutate CPU LRU only; active IDs are sorted."""
    if type(capacity) is not int or capacity < 0:
        raise ValueError('Invalid cache capacity')
    if active != sorted(set(active)) or any(type(i) is not int or i < 0 for i in active):
        raise ValueError('Expected sorted unique nonnegative row IDs')
    if len(state)>capacity or len(set(state.values()))!=len(state):
        raise ValueError('Invalid residency map')
    hits=[(r,state[r]) for r in active if r in state]
    misses=[r for r in active if r not in state]
    # All hits are consumed before any slot is overwritten by the caller.
    for r,_ in hits: state.move_to_end(r)
    inserts=[]
    if capacity:
        for r in misses:
            slot=len(state) if len(state)<capacity else state.popitem(last=False)[1]
            state[r]=slot; inserts.append((r,slot))
    # A call can contain more misses than cache slots. Copy only the final owner.
    inserts=[(r,s) for r,s in inserts if state.get(r)==s]
    return hits,misses,inserts


class RetainedRows:
    def __init__(self,layers,capacity=1024,device='cuda',record=lambda r:None):
        self.layers=list(layers); self.device=torch.device(device)
        self.cuda=self.device.type=='cuda'; self.record=record; self.capacity=capacity
        if not self.layers or type(capacity) is not int or capacity<0:
            raise ValueError('Invalid layers/capacity')
        first=self.layers[0].fc2.weight
        self.hidden,self.neurons=first.shape; self.dtype=first.dtype
        if self.dtype not in (torch.float32,torch.float16) or any(
            l.fc2.weight.device.type!='cpu' or l.fc2.weight.shape!=first.shape
            or l.fc2.weight.dtype!=self.dtype or l.fc2.bias is None for l in self.layers):
            raise ValueError('Uniform biased CPU-first FP32/FP16 weights required')
        self.itemsize=first.element_size(); self.host=[]; self.original=[]
        for i,l in enumerate(self.layers):
            packed=l.fc2.weight.detach().T.contiguous()
            if not bool(torch.isfinite(packed).all()): raise ValueError('Nonfinite weights')
            self.host.append(packed); self.original.append(l.fc2.forward)
            l.fc2.weight.data=packed.T
            l.fc2.forward=lambda x,i=i:self.forward(i,x)
        self.workspace=torch.zeros((self.hidden,self.neurons),dtype=self.dtype,device=self.device).T
        size=self.neurons*(8+self.hidden*self.itemsize)
        self.packet_host=torch.empty(size,dtype=torch.uint8,pin_memory=self.cuda)
        self.packet_device=torch.empty(size,dtype=torch.uint8,device=self.device)
        self.cache=torch.empty((len(self.layers),capacity,self.hidden),dtype=self.dtype,device=self.device)
        self.maps=[OrderedDict() for _ in self.layers]; self.episode=None

    def begin(self,episode,condition):
        if condition not in ('packet','retained'): raise ValueError('Unknown condition')
        self.episode=episode; self.condition=condition; self.call=0
        for m in self.maps: m.clear()

    def indices(self,values):
        return torch.tensor(values,dtype=torch.int64,device='cpu').to(self.device)

    def synchronize(self):
        if self.cuda: torch.cuda.synchronize()

    @torch.inference_mode()
    def forward(self,index,x):
        if self.episode is None or x.ndim!=2 or x.shape[1]!=self.neurons or x.dtype!=self.dtype:
            raise ValueError('Invalid outgoing projection call')
        self.synchronize(); start=time.perf_counter()
        if not bool(torch.isfinite(x).all().item()): raise ValueError('Nonfinite activation')
        activity=(x!=0).cpu().numpy().copy()
        active=np.flatnonzero(activity.any(axis=0)).tolist()
        hits,misses,inserts=retention_plan(self.maps[index],active,
            self.capacity if self.condition=='retained' else 0)
        selected=time.perf_counter(); index_bytes=0
        if hits:
            rows,slots=zip(*hits); ri=self.indices(rows); si=self.indices(slots)
            index_bytes+=16*len(hits)
            self.workspace.index_copy_(0,ri,self.cache[index].index_select(0,si))
        n=len(misses); header=8*n; payload=n*self.hidden*self.itemsize; total=header+payload
        if n:
            host_ids=self.packet_host[:header].view(torch.int64)
            host_ids.copy_(torch.tensor(misses,dtype=torch.int64))
            host_rows=self.packet_host[header:total].view(self.dtype).reshape(n,self.hidden)
            torch.index_select(self.host[index],0,host_ids,out=host_rows)
            self.packet_device[:total].copy_(self.packet_host[:total],non_blocking=False)
            ids=self.packet_device[:header].view(torch.int64)
            rows=self.packet_device[header:total].view(self.dtype).reshape(n,self.hidden)
            self.workspace.index_copy_(0,ids,rows)
        if inserts:
            rows,slots=zip(*inserts); ri=self.indices(rows); si=self.indices(slots)
            index_bytes+=16*len(inserts)
            self.cache[index].index_copy_(0,si,self.workspace.index_select(0,ri))
        self.synchronize(); acquired=time.perf_counter()
        y=torch.nn.functional.linear(x,self.workspace.T,self.layers[index].fc2.bias)
        self.synchronize(); finished=time.perf_counter(); self.call+=1
        self.record(dict(episode=self.episode,condition=self.condition,call=self.call,layer=index,
            tokens=len(x),active=active,hits=hits,misses=misses,inserts=inserts,
            residency=list(self.maps[index].items()),activity=np.packbits(activity,axis=1),
            weight_h2d_bytes=payload if self.cuda else 0,
            metadata_h2d_bytes=(header+index_bytes) if self.cuda else 0,
            activity_d2h_bytes=(activity.nbytes+1) if self.cuda else 0,
            started=start,selection_finished=selected,acquisition_finished=acquired,finished=finished))
        return y

    def allocation(self):
        return dict(host_weights_bytes=sum(t.numel()*t.element_size() for t in self.host),
            cache_bytes=self.cache.numel()*self.itemsize,workspace_bytes=self.workspace.numel()*self.itemsize,
            packet_cuda_bytes=self.packet_device.numel(),pinned_bytes=self.packet_host.numel(),
            capacity_rows_per_layer=self.capacity,host_residency_entries_limit=len(self.layers)*self.capacity,
            host_map_bytes_in_process_rss=True,original_layout=self.workspace.T.is_contiguous())
