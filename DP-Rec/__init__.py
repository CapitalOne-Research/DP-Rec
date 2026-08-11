"""
DP-Rec: Dynamic-Patch Recommendation — clean-room implementation.

Modules:
    ops                           — shared primitives (RoPE, masks, FFN, attention, blocks)
    behavioral_boundary_detector  — BehavioralBoundaryDetector + BehavioralBoundaryModel
    simple_patchers               — FixedCountPatcher + RandomPatcher (parameter-free)
    temporal_local_encoder        — TemporalLocalEncoder
    temporal_latent_transformer   — TemporalLatentTransformer
    local_decoder                 — LocalDecoder
    dprec                         — DPREC (top-level model)
"""

from .behavioral_boundary_detector import BehavioralBoundaryDetector, BehavioralBoundaryModel
from .simple_patchers import FixedCountPatcher, RandomPatcher
from .temporal_local_encoder import TemporalLocalEncoder
from .temporal_latent_transformer import TemporalLatentTransformer
from .local_decoder import LocalDecoder
from .dprec import DPREC

__all__ = [
    "BehavioralBoundaryDetector",
    "BehavioralBoundaryModel",
    "FixedCountPatcher",
    "RandomPatcher",
    "TemporalLocalEncoder",
    "TemporalLatentTransformer",
    "LocalDecoder",
    "DPREC",
]
