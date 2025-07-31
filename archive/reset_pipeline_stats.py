def reset_pipeline_state(self) -> None:
    """
    Resets pipeline status to initial state for new runs.
    """
    self._pipeline_status = "initialized"
    self._current_step = "none"
    self._pipeline_error = None
    self._logger.info("Pipeline state reset")