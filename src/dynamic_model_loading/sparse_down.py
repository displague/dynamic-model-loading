"""Exact-zero discovery followed by physical contiguous outgoing-weight loads.

The first projection is always computed by the original model. Unloaded rows in
the shared workspace are finite stale values, NEVER a signal of inactivity: a
row is irrelevant only when the observed activation is exactly zero for every
token in the current call. The matrix multiplication remains dense PyTorch.
"""
import time
import numpy as np
import torch


def active_extents(activity, width=128):
    """Union exact observed activity across this call, then coalesce adjacent pages."""
    a = np.asarray(activity)
    if a.dtype != bool or a.ndim != 2 or not a.size or type(width) is not int or width <= 0:
        raise ValueError('Expected a nonempty boolean activity matrix and positive page width')
    n = a.shape[1]
    pages = [i for i, start in enumerate(range(0, n, width)) if a[:,start:start+width].any()]
    extents = []
    for page in pages:
        start, stop = page*width, min((page+1)*width,n)
        if extents and extents[-1][1] == start:
            extents[-1][1] = stop
        else:
            extents.append([start,stop])
    return pages, extents


class SparseDown:
    """One serialized workspace/staging pair shared by all host-backed fc2 layers."""
    def __init__(self, layers, record=lambda row:None, width=128, device='cuda'):
        self.layers, self.record, self.width = list(layers), record, width
        if not self.layers or type(width) is not int or width <= 0:
            raise ValueError('Missing layers or invalid width')
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.host = []
        self.original = []
        self.condition = 'stream'
        self.episode = None
        self.call = 0
        self.copy_h2d = self.copy_d2h = self.loads = 0
        first = self.layers[0].fc2.weight
        self.hidden, self.neurons = first.shape
        if any(layer.fc2.weight.shape != first.shape or layer.fc2.weight.dtype != torch.float32
               or layer.fc2.bias is None for layer in self.layers):
            raise ValueError('Only uniform biased FP32 output projections are supported')
        self.construction_d2h = 0
        for layer in self.layers:
            source = layer.fc2
            # Pack once as contiguous neuron rows; fc2.weight aliases the packed
            # host storage through its transposed view. No hidden backup persists.
            packed = source.weight.detach().T.contiguous().cpu()
            if source.weight.is_cuda:
                self.construction_d2h += packed.numel()*packed.element_size()
            if not bool(torch.isfinite(packed).all()):
                raise ValueError('Nonfinite host weights')
            source.weight.data = packed.T
            self.host.append(packed)
            self.original.append(source.forward)
        self.workspace = torch.zeros((self.neurons,self.hidden),dtype=torch.float32,device=self.device)
        self.staging = torch.empty((self.neurons,self.hidden),dtype=torch.float32,pin_memory=self.cuda)
        for i, layer in enumerate(self.layers):
            layer.fc2.forward = lambda x,i=i:self.forward(i,x)

    def begin(self, episode, condition):
        if condition not in ('stream','sparse'):
            raise ValueError('Unknown acquisition condition')
        self.episode, self.condition = episode, condition
        self.call = 0
        self.copy_h2d = self.copy_d2h = self.loads = 0

    @torch.inference_mode()
    def forward(self, index, x):
        if self.episode is None or x.ndim != 2 or x.shape[1] != self.neurons or x.dtype != torch.float32:
            raise ValueError('Invalid physical output-projection call')
        if self.cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        # Exact activity discovery is candidate work, not charged to the stream control.
        activity = None
        copies_d2h = 0
        if self.condition == 'sparse':
            valid = bool(torch.isfinite(x).all().item())
            copies_d2h += int(self.cuda)
            if not valid:
                raise ValueError('Nonfinite activations cannot certify an omitted row')
            activity = (x != 0).cpu().numpy().copy()
            copies_d2h += activity.nbytes if self.cuda else 0
            pages, extents = active_extents(activity,self.width)
        else:
            pages = list(range((self.neurons+self.width-1)//self.width))
            extents = [[0,self.neurons]]
        selection_finished = time.perf_counter()
        payload = 0
        for start, stop in extents:
            count = stop-start
            stage = self.staging[:count]
            stage.copy_(self.host[index][start:stop])
            # Blocking copy ensures staging is not overwritten while in flight.
            self.workspace[start:stop].copy_(stage,non_blocking=False)
            payload += count*self.hidden*4
        if self.cuda:
            torch.cuda.synchronize()
        acquisition_finished = time.perf_counter()
        result = x @ self.workspace + self.layers[index].fc2.bias
        if self.cuda:
            torch.cuda.synchronize()
        compute_finished = time.perf_counter()
        self.copy_h2d += payload if self.cuda else 0
        self.copy_d2h += copies_d2h
        self.loads += len(extents)
        self.call += 1
        self.record(dict(episode=self.episode,condition=self.condition,call=self.call,layer=index,
            tokens=len(x),pages=pages,extents=extents,weight_h2d_bytes=payload if self.cuda else 0,
            activity_d2h_bytes=copies_d2h,activity=None if activity is None else np.packbits(activity,axis=1),
            started=started,selection_finished=selection_finished,
            acquisition_finished=acquisition_finished,compute_finished=compute_finished))
        return result

    def restore(self):
        """Restore original dense methods and contiguous device weight layout."""
        transferred = 0
        for layer, packed, forward in zip(self.layers,self.host,self.original,strict=True):
            layer.fc2.forward = forward
            layer.fc2.weight.data = packed.T.contiguous().to(self.device)
            transferred += packed.numel()*packed.element_size() if self.cuda else 0
        return transferred

    def allocation(self):
        return dict(host_down_bytes=sum(a.numel()*a.element_size() for a in self.host),
            workspace_bytes=self.workspace.numel()*self.workspace.element_size(),
            pinned_staging_bytes=self.staging.numel()*self.staging.element_size() if self.cuda else 0,
            construction_d2h_bytes=self.construction_d2h,
            host_weight_aliases=all(layer.fc2.weight.untyped_storage().data_ptr()==a.untyped_storage().data_ptr()
                                  for layer,a in zip(self.layers,self.host,strict=True)))
