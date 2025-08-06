from typing import Callable, Dict, Type

from src.pipeline.pipeline_moduls.evaluation.base.metric_base import MetricBase


class MetricRegistry:
    """
    Registry for MetricBase subclasses.

    Allows registering metric classes with a unique string key,
    retrieving metric classes by name, and listing all registered metrics.
    """

    _registry: Dict[str, Type[MetricBase]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[MetricBase]], Type[MetricBase]]:
        """
        Decorator to register a metric class under a given name.

        Args:
            name: Unique name to register the metric under.

        Returns:
            Decorator that registers the metric class.
        """

        def decorator(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
            if name in cls._registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            cls._registry[name] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def get_metric_cls(cls, name: str) -> Type[MetricBase]:
        """
        Retrieve the metric class registered under the given name.

        Args:
            name: Name of the registered metric.

        Raises:
            ValueError: If the metric is not registered.

        Returns:
            The metric class associated with the name.
        """
        if name not in cls._registry:
            available = cls.list_metrics()
            raise ValueError(
                f"Metric '{name}' is not registered. Available metrics: "
                f" {list(available.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_metrics(cls) -> Dict[str, Type[MetricBase]]:
        """
        Returns a copy of the current registry mapping.

        Returns:
            Dictionary mapping metric names to their classes.
        """
        return cls._registry.copy()
