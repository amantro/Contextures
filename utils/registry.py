# utils/registry.py

"""
Decorator-based registry
"""

from collections import defaultdict
from typing import Callable, Dict, Type, Any

_REGISTRIES: Dict[str, Dict[str, Any]] = defaultdict(dict)

def _register(category: str, name: str) -> Callable[[Any], Any]:
    def decorator(obj: Any) -> Any:
        name_lc = name.lower()
        if name_lc in _REGISTRIES[category]:
            raise KeyError(f"{category} '{name}' already registered")
        _REGISTRIES[category][name_lc] = obj
        return obj
    
    return decorator

def _get(category: str, name: str) -> Any:
    try:
        return _REGISTRIES[category][name.lower()]
    except KeyError as err:
        raise KeyError(f"{category} '{name}' is not registered") from err

# feature transforms
def register_transform(name: str) -> Callable[[Type], Type]:
    return _register('transform', name)

def get_transform(name: str) -> Any:
    return _get('transform', name)

def list_transforms() -> list[str]:
    return list(_REGISTRIES['transform'])

# encoders
def register_encoder(name: str):
    return _register('encoder', name)

def get_encoder(name: str):
    return _get('encoder', name)

def list_encoders() -> list[str]:
    return list(_REGISTRIES['encoder'])

# contexts
def register_context(name: str):
    return _register('context', name)

def get_context(name: str):
    return _get('context', name)

def list_contexts() -> list[str]:
    return list(_REGISTRIES['context'])


# losses
def register_loss(name: str):
    return _register('loss', name)

def get_loss(name: str):
    return _get('loss', name)

def list_losses() -> list[str]:
    return list(_REGISTRIES['loss'])

try:
    import encoders
except ImportError:
    pass

try:
    import contexts
except ImportError:
    pass

try:
    import losses
except ImportError:
    pass
