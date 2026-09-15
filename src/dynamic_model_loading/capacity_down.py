"""CPU-first outgoing weights with exact packets and a direct dense-copy control."""
import time
import torch
from .row_packet import OriginalRowPacket


def move_except_down(model,layers,device='cuda',check=lambda:None):
    """Move unique original Parameter objects; tied embeddings remain tied."""
    excluded={id(layer.fc2.weight) for layer in layers}
    if any(p.is_cuda for p in model.parameters()):
        raise ValueError('Capacity construction requires CPU-only parameters')
    moved=0
    for p in model.parameters():
        if id(p) in excluded: continue
        p.data=p.data.to(device); moved+=p.numel()*p.element_size(); check()
    for b in model.buffers():
        b.data=b.data.to(device); moved+=b.numel()*b.element_size(); check()
    if any(layer.fc2.weight.device.type!='cpu' for layer in layers):
        raise ValueError('Host weights moved into device residency')
    return moved


class CapacityDown(OriginalRowPacket):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Extra host layout is deliberate: give dense streaming a contiguous,
        # original-layout H2D path without per-call transpose/device temporary.
        self.host_linear=[a.T.contiguous() for a in self.host]

    @torch.inference_mode()
    def forward(self,index,x):
        if self.condition!='stream': return super().forward(index,x)
        if self.episode is None or x.ndim!=2 or x.shape[1]!=self.neurons or x.dtype!=torch.float32:
            raise ValueError('Invalid direct-stream call')
        if self.cuda: torch.cuda.synchronize()
        started=time.perf_counter(); selection_finished=time.perf_counter()
        stage=self.staging.view(self.hidden,self.neurons)
        stage.copy_(self.host_linear[index])
        self.workspace.T.copy_(stage,non_blocking=False)
        if self.cuda: torch.cuda.synchronize()
        acquisition_finished=time.perf_counter()
        result=self.compute(index,x)
        if self.cuda: torch.cuda.synchronize()
        compute_finished=time.perf_counter()
        payload=self.hidden*self.neurons*4 if self.cuda else 0
        self.call+=1; self.copy_h2d+=payload; self.loads+=1
        self.record(dict(episode=self.episode,condition='stream',call=self.call,layer=index,
            tokens=len(x),pages=list(range((self.neurons+self.width-1)//self.width)),
            extents=[[0,self.neurons]],weight_h2d_bytes=payload,activity_d2h_bytes=0,
            activity=None,started=started,selection_finished=selection_finished,
            acquisition_finished=acquisition_finished,compute_finished=compute_finished,
            direct_contiguous_copy=stage.is_contiguous() and self.workspace.T.is_contiguous()))
        return result

    def allocation(self):
        return dict(super().allocation(),
            extra_host_linear_bytes=sum(a.numel()*a.element_size() for a in self.host_linear),
            direct_dense_host_contiguous=all(a.is_contiguous() for a in self.host_linear))
