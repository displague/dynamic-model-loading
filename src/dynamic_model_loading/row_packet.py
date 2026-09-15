"""Exact-active outgoing rows plus their indices in one physical transfer packet."""
import time
import numpy as np
import torch
from .sparse_down import SparseDown


class RowPacket(SparseDown):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep the stream control's staging and charge both buffers explicitly.
        size=self.neurons*(8+self.hidden*4)
        self.packet_host=torch.empty(size,dtype=torch.uint8,pin_memory=self.cuda)
        self.packet_device=torch.empty(size,dtype=torch.uint8,device=self.device)

    def begin(self,episode,condition):
        super().begin(episode,'sparse' if condition=='packet' else condition)
        self.condition=condition

    @torch.inference_mode()
    def forward(self,index,x):
        if self.condition!='packet': return super().forward(index,x)
        if self.episode is None or x.ndim!=2 or x.shape[1]!=self.neurons or x.dtype!=torch.float32:
            raise ValueError('Invalid packet projection call')
        if self.cuda: torch.cuda.synchronize()
        started=time.perf_counter()
        if not bool(torch.isfinite(x).all().item()): raise ValueError('Nonfinite activity')
        activity=(x!=0).cpu().numpy().copy()
        rows=np.flatnonzero(activity.any(axis=0)).astype(np.int64)
        count=len(rows); header=count*8; payload=count*self.hidden*4; total=header+payload
        selection_finished=time.perf_counter()
        if count:
            host_ids=self.packet_host[:header].view(torch.int64)
            host_rows=self.packet_host[header:total].view(torch.float32).reshape(count,self.hidden)
            host_ids.copy_(torch.from_numpy(rows))
            torch.index_select(self.host[index],0,host_ids,out=host_rows)
        packing_finished=time.perf_counter()
        if count: self.packet_device[:total].copy_(self.packet_host[:total],non_blocking=False)
        if self.cuda: torch.cuda.synchronize()
        transfer_finished=time.perf_counter()
        if count:
            device_ids=self.packet_device[:header].view(torch.int64)
            device_rows=self.packet_device[header:total].view(torch.float32).reshape(count,self.hidden)
            self.workspace.index_copy_(0,device_ids,device_rows)
        if self.cuda: torch.cuda.synchronize()
        acquisition_finished=time.perf_counter()
        result=self.compute(index,x)
        if self.cuda: torch.cuda.synchronize()
        compute_finished=time.perf_counter()
        d2h=(activity.nbytes+1) if self.cuda else 0
        self.copy_h2d+=total if self.cuda else 0; self.copy_d2h+=d2h
        self.loads+=int(count>0); self.call+=1
        self.record(dict(episode=self.episode,condition='packet',call=self.call,layer=index,
            tokens=len(x),rows=rows.tolist(),packets=int(count>0),weight_h2d_bytes=payload if self.cuda else 0,
            metadata_h2d_bytes=header if self.cuda else 0,packet_h2d_bytes=total if self.cuda else 0,
            activity_d2h_bytes=d2h,activity=np.packbits(activity,axis=1),
            started=started,selection_finished=selection_finished,packing_finished=packing_finished,
            transfer_finished=transfer_finished,acquisition_finished=acquisition_finished,
            compute_finished=compute_finished))
        return result

    def allocation(self):
        return dict(super().allocation(),packet_host_bytes=self.packet_host.numel(),
            packet_cuda_bytes=self.packet_device.numel() if self.cuda else 0,
            total_pinned_bytes=self.staging.numel()*4+self.packet_host.numel() if self.cuda else 0)


class OriginalRowPacket(RowPacket):
    """Corrected path preserves the original dense weight layout and bias operator."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs,original_layout=True)
