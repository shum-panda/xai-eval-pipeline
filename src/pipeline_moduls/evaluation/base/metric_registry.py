from typing import Dict, Type

from pipeline_moduls.evaluation.base.metric_base import MetricBase


class MetricRegistry:
    _registry: Dict[str, Type[MetricBase]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_cls: Type[MetricBase]):
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def get_metric_cls(cls, name: str) -> Type[MetricBase]:
        if name not in cls._registry:
            metric_list = MetricRegistry.list_metrics()
            raise ValueError(
                f"Metric '{name}' is not registered."
                f"the available Metrics: {metric_list}"
            )
        return cls._registry[name]

    @classmethod
    def list_metrics(cls) -> Dict[str, Type[MetricBase]]:
        return cls._registry.copy()
