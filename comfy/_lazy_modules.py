"""
_lazy_modules.py — Dehydrate 惰性模块代理

ComfyDL 脱水模式下，扩散生成实现层（comfy.ldm / comfy.text_encoders / comfy.taesd /
comfy.k_diffusion / comfy.lora 等）将被删除。为了让保留的核心模块（model_patcher / hooks /
sd 等）在 import 时不触发生成链加载，同时保留"数据结构协议"（类型可复用），这里提供：

  - install_lazy_submodules(): 把 comfy 包下未加载的生成子模块注册为惰性属性代理。
    实际访问时才触发 import；对应文件已被删除时抛出明确的 ImportError。
  - _LazyModule: 惰性模块代理（属性访问时加载）。
  - _LazyCallable: 惰性类引用（用于 `AutoencoderKL(...)` 这类类实例化调用点，
    无需改写函数体内几十处调用）。

用法（在 model_patcher / hooks / sd 顶部）：
    import comfy
    from comfy._lazy_modules import install_lazy_submodules
    install_lazy_submodules()
    之后代码中 `comfy.lora.calculate_weight(...)` 等访问自动惰性化。
"""
import importlib
import sys

# comfy 包下应惰性化的生成实现子模块（脱水后将被删除）
LAZY_SUBMODULES = (
    "ldm",
    "text_encoders",
    "taesd",
    "k_diffusion",
    "extra_samplers",
    "image_encoders",
    "audio_encoders",
    "weight_adapter",
    "t2i_adapter",
    "cldm",
    "background_removal",
    # 生成单文件
    "lora",
    "lora_convert",
    "diffusers_convert",
    "model_detection",
    "supported_models",
    "supported_models_base",
    "model_base",
    "model_sampling",
    "latent_formats",
    "conds",
    "context_windows",
    "clip_vision",
    "clip_model",
    "gligen",
    "sd1_clip",
    "sdxl_clip",
    "pixel_space_convert",
    "samplers",
    "sample",
    "sampler_helpers",
    "controlnet",
    "diffusers_load",
    "rmsnorm",
    "bg_removal_model",
)

_INSTALLED = False


class _LazyModule:
    """属性访问时惰性 import 的模块代理。

    解析策略：
      1. 优先尝试 import <name>.<item>（item 可能是子模块/子包）；
      2. 失败则加载 <name> 本身并从其中取属性 item。
    被访问的生成模块已被删除时，import 抛 ModuleNotFoundError，报错清晰。
    """

    def __init__(self, name):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_loaded", None)

    def _load(self):
        mod = object.__getattribute__(self, "_loaded")
        if mod is None:
            mod = importlib.import_module(object.__getattribute__(self, "_name"))
            object.__setattr__(self, "_loaded", mod)
        return mod

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        name = object.__getattribute__(self, "_name")
        try:
            sub = importlib.import_module(f"{name}.{item}")
        except ImportError:
            mod = self._load()
            return getattr(mod, item)
        setattr(self, item, sub)
        return sub

    def __call__(self, *args, **kwargs):
        raise TypeError(f"{object.__getattribute__(self, '_name')} is a module, not callable")

    def __repr__(self):
        return f"<lazy module '{object.__getattribute__(self, '_name')}'>"


class _LazyCallable:
    """延迟到调用时才解析的类引用（覆盖 `AutoencoderKL(...)` 类实例化调用点）。"""

    def __init__(self, module_name, attr_name):
        self._module_name = module_name
        self._attr_name = attr_name
        self._resolved = None

    def _resolve(self):
        if self._resolved is None:
            mod = importlib.import_module(self._module_name)
            self._resolved = getattr(mod, self._attr_name)
        return self._resolved

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)


def install_lazy_submodules(pkg_name="comfy"):
    """把 comfy 包下未加载的生成子模块注册为惰性属性（幂等，可多次调用）。"""
    global _INSTALLED
    if _INSTALLED:
        return
    import comfy
    for name in LAZY_SUBMODULES:
        if name in sys.modules:
            continue  # 已被真实加载，保持真实模块
        if hasattr(comfy, name):
            continue  # 已有属性（真实模块或已注册代理），不覆盖
        setattr(comfy, name, _LazyModule(f"{pkg_name}.{name}"))
    _INSTALLED = True
