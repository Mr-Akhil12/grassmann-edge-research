"""Geometric sequence modeling research pilot, not a paper replication."""
from .models import GrassmannMix, LocalMix, TransformerMix, TinyLM, wedge, count_parameters
from .synthetic import delayed_copy_batch

model_parameters = count_parameters
__all__ = ["GrassmannMix", "LocalMix", "TransformerMix", "TinyLM", "wedge", "model_parameters", "delayed_copy_batch"]
