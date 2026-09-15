"""CPU-first construction and independent accounting for the larger FP16 screen."""
import time
import torch
from .capacity_down import move_except_down
from .experiment import cuda_memory
from .debt_analysis import demand


TOTAL=5303193600
OUTGOING=1677721600
NON_OUTGOING=3625472000
LIMIT=4800*2**20


def construct(model,layers,bank_type,cfg,resources,setup_started):
    demand(all(p.device.type=='cpu' and p.dtype==torch.float16 for p in model.parameters()),'Not CPU-first FP16')
    before=cuda_memory(torch.device('cuda:0'))
    demand(all(before[k]==0 for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')),'Hidden CUDA preload')
    total=sum(p.numel()*p.element_size() for p in model.parameters())
    demand(total==TOTAL and sum(l.fc2.weight.numel()*2 for l in layers)==OUTGOING,'Architecture byte count changed')
    bank=bank_type(layers,capacity=cfg['cache_rows'])
    def check():
        resources.check()
        demand(cuda_memory(torch.device('cuda:0'))['peak_reserved_bytes']<=LIMIT,'Construction allocator limit exceeded')
    moved=move_except_down(model,layers,check=check); torch.cuda.synchronize(); resources.boundary()
    allocation=dict(bank.allocation(),cpu_first=True,before_candidate_cuda=before,
        cpu_loaded_parameter_bytes=total,construction_h2d_bytes=moved,construction_d2h_bytes=0,
        candidate_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda),
        candidate_cuda=cuda_memory(torch.device('cuda:0')),candidate_setup_seconds=time.perf_counter()-setup_started,
        reference_kind='dense_contiguous_fp16_stream',full_gpu_model_loaded=False,
        allocator_fraction=torch.cuda.memory.get_per_process_memory_fraction(),
        tied_embeddings=model.lm_head.weight is model.model.decoder.embed_tokens.weight)
    demand(moved==allocation['candidate_parameter_bytes']==NON_OUTGOING,'Physical construction bytes changed')
    return bank,allocation


def audit_allocation(a,total_gpu_bytes):
    demand(a['cpu_first'] is True and a['full_gpu_model_loaded'] is False and a['tied_embeddings'] is True,'Cold construction not qualified')
    demand(all(a['before_candidate_cuda'][k]==0 for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')),'Hidden initial residency')
    demand(a['cpu_loaded_parameter_bytes']==TOTAL and a['construction_h2d_bytes']==a['candidate_parameter_bytes']==NON_OUTGOING
        and a['construction_d2h_bytes']==0 and a['host_weights_bytes']==a['extra_host_linear_bytes']==OUTGOING,'Scale allocation changed')
    demand(a['parameter_dtype']=='torch.float16' and a['direct_host_contiguous'] is True
        and a['reference_kind']=='dense_contiguous_fp16_stream' and a['allocator_fraction']==LIMIT/total_gpu_bytes,'Scale representation changed')
    demand(a['candidate_cuda']['peak_reserved_bytes']<=LIMIT and TOTAL>LIMIT,'Capacity evidence failed')
