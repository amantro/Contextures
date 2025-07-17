# encoders/__init__.py

"""
Expose all encoder classes and register them on import

Usage: 
from utils.registry import get_encoder
enc_cls = get_encoder('MLPEncoder')
"""

from .mlp import MLPEncoder