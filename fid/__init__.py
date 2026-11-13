"""Dead Stock Detector.

Find stock that is quietly going dead, and tell it apart from stock that is merely paused,
seasonal or slow.

Quick start::

    from fid import analyze, Config
    result = analyze(movements, sku_master, Config()).result
"""
from .config import CLASSES, Config
from .pipeline import Analysis, analyze, sku_timeline

__all__ = ["analyze", "Analysis", "Config", "CLASSES", "sku_timeline"]
__version__ = "1.0.0"
