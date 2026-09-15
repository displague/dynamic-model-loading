"""Change acquisition grain by token batch without changing arithmetic or KV."""
import torch
from .precision_rows import PrecisionRows


class PhaseRows(PrecisionRows):
    def begin(self,episode,condition):
        if condition not in ('stream','packet','phase'):
            raise ValueError('Unknown phase condition')
        super().begin(episode,'packet' if condition=='phase' else condition)
        self.condition=condition

    @torch.inference_mode()
    def forward(self,index,x):
        if self.condition!='phase': return super().forward(index,x)
        physical='stream' if len(x)>1 else 'packet'
        original_record=self.record
        def record(row):
            row['physical_condition']=physical
            row['condition']='phase'
            original_record(row)
        self.record=record; self.condition=physical
        try:
            return super().forward(index,x)
        finally:
            self.record=original_record; self.condition='phase'
