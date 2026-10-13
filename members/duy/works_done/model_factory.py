"""Small model entry point for Huy's future shared trainer."""

from importlib import import_module

from torch import nn

from .unet import UNet


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    if name == "unet":
        if kwargs.pop("deep_supervision", False):
            raise ValueError("Plain U-Net has no deep supervision; use Phuc's U-Net++ for that experiment")
        return UNet(**kwargs)
    if name == "unetplusplus":
        module_name = "members.phuc.works_done.unetplusplus"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            # Do not hide a missing dependency imported inside Phuc's module.
            if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
                raise ModuleNotFoundError(
                    "U-Net++ is not delivered yet. Phuc must provide "
                    "members/phuc/works_done/unetplusplus.py with UNetPlusPlus."
                ) from exc
            raise
        constructor = getattr(module, "UNetPlusPlus", None)
        if constructor is None:
            raise AttributeError(f"{module_name} must expose UNetPlusPlus; agree its constructor with Phuc")
        return constructor(**kwargs)
    raise ValueError(f"Unknown model {name!r}; expected 'unet' or 'unetplusplus'")

