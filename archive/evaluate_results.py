def evaluate_results(
        self, results: List[XAIExplanationResult]
) -> EvaluationSummary:
    """
    Evaluates a batch of explanation results and logs metrics to MLflow.
    """
    self._logger.info("Calculating evaluation metrics...")

    individual_metrics: List[Any] = []
    correct_predictions = 0
    total_processing_time = 0.0

    self._logger.info(
        f"Processing {len(results)} results for individual metrics..."
    )

    for i, result in enumerate(results):
        if result.prediction_correct is not None and result.prediction_correct:
            correct_predictions += 1

        total_processing_time += result.processing_time

        metrics = self._evaluator.evaluate_single_result(result)
        individual_metrics.append(metrics)

        if (i + 1) % 10 == 0:
            self._logger.info(
                f"Processed {i + 1}/{len(results)} individual metrics"
            )

    self._individual_metrics = individual_metrics

    summary = self._evaluator.create_summary_from_individual_metrics(
        results=results,
        individual_metrics=individual_metrics,
        correct_predictions=correct_predictions,
        total_processing_time=total_processing_time,
    )

    self._logger.info("Evaluation metrics calculation finished!")

    for key, value in summary.to_dict().items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

    return summary