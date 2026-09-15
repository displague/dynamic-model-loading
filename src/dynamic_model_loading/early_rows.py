"""Causal prior-row prefetch before fc1, with exact observed demand completion."""
import time
import numpy as np
import torch
from .retained_rows import RetainedRows


def completion_rows(previous,active,budget=1024):
    if type(budget) is not int or budget<0: raise ValueError('Invalid prefetch budget')
    predicted=previous[:budget]
    missing=sorted(set(active)-set(predicted))
    return predicted,missing


class EarlyRows(RetainedRows):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.capacity!=0: raise ValueError('Early packets have no retained weight cache')
        self.budget=min(1024,self.neurons)
        size=self.budget*(8+self.hidden*self.itemsize)
        self.early_host=torch.empty(size,dtype=torch.uint8,pin_memory=self.cuda)
        self.early_device=torch.empty(size,dtype=torch.uint8,device=self.device)
        self.copy_stream=torch.cuda.Stream() if self.cuda else None
        self.previous=[[] for _ in self.layers]; self.pending={}; self.pre_times={}
        self.hooks=[]
        for i,l in enumerate(self.layers):
            self.hooks.append(l.fc1.register_forward_pre_hook(lambda m,args,i=i:self.before_fc1(i)))
            self.hooks.append(l.fc1.register_forward_hook(lambda m,args,out,i=i:self.after_fc1(i)))

    def begin(self,episode,condition):
        if condition not in ('packet','late','early'): raise ValueError('Unknown anticipation condition')
        if self.pending: raise ValueError('Cannot reset an unconsumed prefetch')
        super().begin(episode,'packet'); self.condition=condition
        self.previous=[[] for _ in self.layers]; self.pre_times={}

    def before_fc1(self,index):
        if self.episode is None: raise ValueError('Begin an episode before model execution')
        if self.condition=='packet': return
        if index in self.pending or index in self.pre_times: raise ValueError('Previous forecast was not consumed')
        self.pre_times[index]=time.perf_counter()
        if self.condition=='early':
            self.prepare(index)
            if self.cuda and self.pending[index]['predicted']:
                self.pending[index]['fc1_start'].record(torch.cuda.current_stream())

    def after_fc1(self,index):
        if index in self.pending and self.cuda and self.pending[index]['predicted']:
            self.pending[index]['fc1_end'].record(torch.cuda.current_stream())

    def prepare(self,index):
        predicted=self.previous[index][:self.budget]; n=len(predicted)
        row=dict(predicted=predicted,copy_seconds=0.,overlap_ms=0.,copy_ms=0.,wait_seconds=0.,
            copy_start_ms=0.,copy_end_ms=0.,fc1_start_ms=0.,fc1_end_ms=0.)
        self.pending[index]=row
        if not n: return
        header=n*8; total=header+n*self.hidden*self.itemsize
        ids=self.early_host[:header].view(torch.int64); ids.copy_(torch.tensor(predicted,dtype=torch.int64))
        values=self.early_host[header:total].view(self.dtype).reshape(n,self.hidden)
        torch.index_select(self.host[index],0,ids,out=values)
        begin=time.perf_counter()
        if self.cuda:
            row.update({k:torch.cuda.Event(enable_timing=True) for k in ('anchor','copy_start','copy_end','fc1_start','fc1_end')})
            row['anchor'].record(torch.cuda.current_stream())
            self.copy_stream.wait_event(row['anchor'])
            with torch.cuda.stream(self.copy_stream):
                row['copy_start'].record(); self.early_device[:total].copy_(self.early_host[:total],non_blocking=True)
                row['copy_end'].record()
            # For the late control fc1 is already complete; use zero interval.
            if self.condition=='late': row['fc1_end'].record(torch.cuda.current_stream())
        else: self.early_device[:total].copy_(self.early_host[:total])
        row['copy_seconds']=time.perf_counter()-begin

    @torch.inference_mode()
    def forward(self,index,x):
        if self.condition=='packet':
            return super().forward(index,x)
        if self.episode is None or x.ndim!=2 or x.shape[1]!=self.neurons or x.dtype!=self.dtype:
            raise ValueError('Invalid early projection call')
        if index not in self.pre_times: raise ValueError('fc1 hook did not execute')
        if self.cuda: torch.cuda.current_stream().synchronize()
        start=self.pre_times.pop(index)
        if self.condition=='late': self.prepare(index)
        pending=self.pending.pop(index)
        if not bool(torch.isfinite(x).all().item()): raise ValueError('Nonfinite activity')
        activity=(x!=0).cpu().numpy().copy(); active=np.flatnonzero(activity.any(axis=0)).tolist()
        predicted,missing=completion_rows(self.previous[index],active,self.budget)
        if predicted!=pending['predicted']: raise ValueError('Forecast changed after observation')
        selection=time.perf_counter(); n=len(predicted)
        if n:
            if self.cuda:
                wait=time.perf_counter(); pending['copy_end'].synchronize()
                pending['wait_seconds']=time.perf_counter()-wait
                a=pending['anchor'].elapsed_time(pending['copy_start'])
                b=pending['anchor'].elapsed_time(pending['copy_end'])
                end=pending['anchor'].elapsed_time(pending['fc1_end'])
                begin=pending['anchor'].elapsed_time(pending['fc1_start']) if self.condition=='early' else end
                pending.update(copy_start_ms=a,copy_end_ms=b,fc1_start_ms=begin,fc1_end_ms=end)
                pending['copy_ms']=b-a
                pending['overlap_ms']=max(0.,min(b,end)-max(a,begin)) if self.condition=='early' else 0.
            h=n*8; total=h+n*self.hidden*self.itemsize
            self.workspace.index_copy_(0,self.early_device[:h].view(torch.int64),
                self.early_device[h:total].view(self.dtype).reshape(n,self.hidden))
        m=len(missing); h=m*8; total=h+m*self.hidden*self.itemsize
        if m:
            ids=self.packet_host[:h].view(torch.int64); ids.copy_(torch.tensor(missing,dtype=torch.int64))
            rows=self.packet_host[h:total].view(self.dtype).reshape(m,self.hidden)
            torch.index_select(self.host[index],0,ids,out=rows)
            self.packet_device[:total].copy_(self.packet_host[:total],non_blocking=False)
            self.workspace.index_copy_(0,self.packet_device[:h].view(torch.int64),
                self.packet_device[h:total].view(self.dtype).reshape(m,self.hidden))
        self.synchronize(); acquired=time.perf_counter()
        y=torch.nn.functional.linear(x,self.workspace.T,self.layers[index].fc2.bias)
        self.synchronize(); finished=time.perf_counter(); self.call+=1; self.previous[index]=active
        self.record(dict(episode=self.episode,condition=self.condition,call=self.call,layer=index,tokens=len(x),
            active=active,predicted=predicted,misses=missing,useful_prefetched=len(set(predicted)&set(active)),
            wasted_prefetched=len(set(predicted)-set(active)),activity=np.packbits(activity,axis=1),
            weight_h2d_bytes=(n+m)*self.hidden*self.itemsize if self.cuda else 0,
            metadata_h2d_bytes=(n+m)*8 if self.cuda else 0,activity_d2h_bytes=activity.nbytes+1 if self.cuda else 0,
            copy_ms=pending['copy_ms'],overlap_ms=pending['overlap_ms'],wait_seconds=pending['wait_seconds'],
            event_intervals={k:pending[k] for k in ('copy_start_ms','copy_end_ms','fc1_start_ms','fc1_end_ms')},
            started=start,selection_finished=selection,acquisition_finished=acquired,finished=finished))
        return y

    def allocation(self):
        return dict(super().allocation(),early_cuda_bytes=self.early_device.numel(),
            early_pinned_bytes=self.early_host.numel(),prefetch_rows=self.budget,
            predictor_host_rows_limit=len(self.layers)*self.neurons,retained_weight_cache=False)
