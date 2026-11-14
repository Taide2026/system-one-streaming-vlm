import s1stream.plugins, dataclasses
from s1stream.config import load_config
from s1stream.registry import BACKBONES
from s1stream import types
print([f.name for f in dataclasses.fields(types.Window)], [f.name for f in dataclasses.fields(types.VideoItem)])
c = load_config("configs/viz_floor.yaml"); b = BACKBONES.build(c.backbone); m = b.model.backbone
print(type(m).__name__, [n for n, _ in m.named_children()])
v = m.visual
print(type(v).__name__, [(n, type(x).__name__, len(x) if hasattr(x, "__len__") else "") for n, x in v.named_children()])
blk = v.blocks[0]; print([(n, type(x).__name__) for n, x in blk.named_children()])
lm = m.language_model; print(type(lm).__name__, [(n, type(x).__name__, len(x) if hasattr(x, "__len__") else "") for n, x in lm.named_children()])
import inspect; print(inspect.signature(m.get_image_features)); print(inspect.getsource(type(v).forward)[:2500])
