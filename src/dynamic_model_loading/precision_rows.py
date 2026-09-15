"""Original-layout direct dense streaming control for FP16/FP32 packets."""
import time
import torch
from .retained_rows import RetainedRows


class PrecisionRows(RetainedRows):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.host_linear=[t.T.contiguous() for t in self.host]

    def begin(self,episode,condition):
        if condition not in ('stream','packet'): raise ValueError('Unknown precision condition')
        super().begin(episode,'packet'); self.condition=condition

    @torch.inference_mode()
    def forward(self,index,x):
        if self.condition=='packet': return super().forward(index,x)
        if self.episode is None or x.ndim!=2 or x.shape[1]!=self.neurons or x.dtype!=self.dtype:
            raise ValueError('Invalid precision-stream call')
        self.synchronize(); started=time.perf_counter()
        payload=self.hidden*self.neurons*self.itemsize
        stage=self.packet_host[:payload].view(self.dtype).reshape(self.hidden,self.neurons)
        stage.copy_(self.host_linear[index]); self.workspace.T.copy_(stage,non_blocking=False)
        self.synchronize(); acquired=time.perf_counter()
        y=torch.nn.functional.linear(x,self.workspace.T,self.layers[index].fc2.bias)
        self.synchronize(); finished=time.perf_counter(); self.call+=1
        self.record(dict(episode=self.episode,condition='stream',call=self.call,layer=index,tokens=len(x),
            activity=None,weight_h2d_bytes=payload if self.cuda else 0,metadata_h2d_bytes=0,activity_d2h_bytes=0,
            direct_contiguous_copy=stage.is_contiguous() and self.workspace.T.is_contiguous(),
            started=started,selection_finished=started,acquisition_finished=acquired,finished=finished))
        return y

    def allocation(self):
        return dict(super().allocation(),extra_host_linear_bytes=sum(t.numel()*self.itemsize for t in self.host_linear),
            parameter_dtype=str(self.dtype),direct_host_contiguous=all(t.is_contiguous() for t in self.host_linear))
