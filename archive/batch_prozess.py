def batch_process(
        self,
        dataloader: DataLoader,
        explainer_names: List[str],
        max_batches: Optional[int] = None,
        explainer_kwargs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, List[XAIExplanationResult]]:
    """
    Batch Verarbeitung mit mehreren Explainern

    Args:
        dataloader: ImageNet Dataloader
        explainer_names: Liste der Explainer Namen
        max_batches: Max Anzahl batches
        explainer_kwargs: Kwargs für jeden Explainer

    Returns:
        Results gruppiert nach Explainer
    """
    explainer_kwargs = explainer_kwargs or {}
    results = {}

    for explainer_name in explainer_names:
        self._logger.info(f"Verarbeite mit {explainer_name}...")

        try:
            # Erstelle Explainer
            kwargs = explainer_kwargs.get(explainer_name, {})
            explainer = self.create_explainer(explainer_name, **kwargs)

            # Verarbeite Dataset
            explainer_results = list(
                self.process_dataloader(
                    dataloader=dataloader,
                    explainer=explainer,
                    max_batches=max_batches,
                )
            )
            self.result_manager.add_results("step_1", explainer_results)
            results[explainer_name] = explainer_results

        except Exception as e:
            self._logger.error(f"Fehler bei {explainer_name}: {e}")
            results[explainer_name] = []

    return results