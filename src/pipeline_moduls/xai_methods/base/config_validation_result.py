from dataclasses import dataclass, field

from pipeline_moduls.xai_methods.base.validation_result import ValidationResult


@dataclass
class ConfigValidationResult:
    """Ergebnis der Config-Validierung"""

    status: ValidationResult
    message: str
    missing_params: list = field(default_factory=list)
    invalid_params: list = field(default_factory=list)
    defaults_used: dict = field(default_factory=dict)
