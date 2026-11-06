from ..config import Settings
from .base import Providers


def build_providers(s: Settings) -> Providers:
    if s.providers == "fake":
        from . import fake
        return fake.build()
    from . import real
    return real.build(s)
