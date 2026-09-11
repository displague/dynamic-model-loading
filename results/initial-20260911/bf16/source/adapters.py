"""Views of existing FFNs; never replace the model's attention or recurrent blocks."""

from torch.nn import functional as F

from .ffn import dimensions


class FFNView:
    def __init__(self, source, names, act_fn):
        self.source = source
        self.gate_proj, self.up_proj, self.down_proj = (getattr(source, name) for name in names)
        self.act_fn = act_fn
        dimensions(self)

    def __call__(self, x):
        return self.source(x)

    def parameters(self):
        return self.source.parameters()

    def register_forward_pre_hook(self, callback):
        return self.source.register_forward_pre_hook(callback)


def extract_ffns(model):
    kind = model.config.model_type
    if kind in ("qwen2", "granite"):
        if model.config.hidden_act != "silu":
            raise ValueError("Only SwiGLU activation is supported")
        result = [FFNView(layer.mlp, ("gate_proj", "up_proj", "down_proj"), layer.mlp.act_fn)
                  for layer in model.model.layers]
    elif kind == "lfm2":
        result = [FFNView(layer.feed_forward, ("w1", "w3", "w2"), F.silu)
                  for layer in model.model.layers]
    else:
        raise ValueError(f"Unsupported model type: {kind}")
    if not result:
        raise ValueError("No FFNs found")
    return result
