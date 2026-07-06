"""Package-wide constants.

These are hard-coded on purpose (no CLI flags) to keep runs consistent.
"""

# Periodically release cached CUDA/Python memory.
MEMORY_CLEANUP_EVERY_EPOCHS: int = 50

# Fixed PosePred model initialization policy used for all runs.
STANDARD_WEIGHT_INIT_POLICY: str = "transformer_standard_v1"
# Small normal std used for learned token/position parameters.
TRANSFORMER_PARAM_INIT_STD: float = 0.02
