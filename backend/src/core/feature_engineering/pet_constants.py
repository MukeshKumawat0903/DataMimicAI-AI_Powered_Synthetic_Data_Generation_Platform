"""
PET (Privacy-Enhanced Technology) constants used across backend components.
Centralizes supported PET names to avoid divergence between suggester and API.
"""

# All supported PET transformations. Keep lowercase strings here.
PET_TYPES = [
    "hash",
    "mask",
    "redact",
    "generalize",
    "suppress",
    "bin",
]

# Convenience set for quick membership checks
PET_TYPES_SET = set(PET_TYPES)


def is_pet(transformation: str) -> bool:
    """Return True if the given transformation name corresponds to a known PET.

    Normalizes case and trims whitespace.
    """
    if not transformation:
        return False
    return transformation.strip().lower() in PET_TYPES_SET
