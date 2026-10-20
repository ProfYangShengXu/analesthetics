"""analesthetics — 绘画质量的量化诊断。

把「这张画好不好」拆成可以测量的量：先出读数，再让模型解释读数。
"""
from .core import Analysis, Observation, analyze, ingest, segment

__all__ = ["Analysis", "Observation", "analyze", "ingest", "segment"]
__version__ = "0.1.0"
