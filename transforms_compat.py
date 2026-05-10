from typing import Any, Callable, Iterable, Optional


class FallbackCompose:
    """A lightweight Compose replacement when torchvision import fails."""

    def __init__(self, transforms: Iterable[Callable[[Any], Any]]):
        self.transforms = list(transforms)

    def __call__(self, sample: Any) -> Any:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def render_torchvision_import_error(exc: Exception) -> str:
    return (
        "torchvision import failed; falling back to internal Compose. "
        f"Original error: {repr(exc)}. "
        "If you are on Windows and see 'DLL load failed while importing _imaging', "
        "please reinstall Pillow and torchvision with versions compatible with your "
        "Python/PyTorch environment."
    )


try:
    from torchvision import transforms as _tv_transforms

    Compose = _tv_transforms.Compose
    TORCHVISION_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on runtime env
    Compose = FallbackCompose
    TORCHVISION_IMPORT_ERROR = exc


def build_compose(transforms: Iterable[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    return Compose(transforms)


def format_torchvision_import_error() -> str:
    if TORCHVISION_IMPORT_ERROR is None:
        return ""
    return render_torchvision_import_error(TORCHVISION_IMPORT_ERROR)
