from enum import Enum


class ValidationResult(Enum):
    """Ergebnis der Parameter-Validierung"""

    VALID = "valid"  # Alle Parameter sind korrekt
    MISSING_USING_DEFAULTS = (
        "missing_using_defaults"  # Parameter fehlen, defaults verwendet
    )
    INVALID = "invalid"  # Parameter sind ungültig
