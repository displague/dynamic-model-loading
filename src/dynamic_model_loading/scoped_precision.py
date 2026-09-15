"""Fixed intervention masks over physical two-bit slabs; no learned policy."""
from .progressive_precision import ProgressiveDraft


class ScopedPrecision(ProgressiveDraft):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selected = set()

    def begin_token(self):
        if any(type(i) is not int or not 0 <= i < len(self.layers) for i in self.selected):
            raise ValueError('Invalid physical site')
        self.cache.begin_token(False)

    def forward_one(self,x,layer,activation):
        self.mode = 'q8' if layer in self.selected else 'q4'
        return super().forward_one(x,layer,activation)
