#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLflow wrapper for AFlow experiments.
Simple wrapper that logs parameters, metrics, and artifacts to MLflow.
"""

import mlflow
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any


class MLflowCallback:
    """Simple callback to log metrics to MLflow during optimization."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize MLflow callback.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this specific run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None

    def __enter__(self):
        """Start MLflow run."""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        """Log an artifact (file) to MLflow."""
        mlflow.log_artifact(local_path)

    def log_artifacts(self, local_dir: str):
        """Log all files in a directory as artifacts."""
        mlflow.log_artifacts(local_dir)


def run_experiment_with_mlflow(
    experiment_id: str,
    dataset: str,
    exec_model: str,
    opt_model: str,
    max_rounds: int,
    sample: int,
    validation_rounds: int,
    optimizer_instance,
    description: str = "",
    tags: Optional[list] = None,
):
    """
    Run AFlow optimization with MLflow tracking.

    Args:
        experiment_id: Unique ID for this experiment
        dataset: Dataset name (GSM8K, MATH, etc.)
        exec_model: Execution model name
        opt_model: Optimizer model name
        max_rounds: Maximum optimization rounds
        sample: Sample size
        validation_rounds: Number of validation rounds
        optimizer_instance: Configured Optimizer instance
        description: Experiment description
        tags: Optional list of tags
    """
    with MLflowCallback(
        experiment_name=f"aflow_{dataset.lower()}",
        run_name=experiment_id
    ) as mlflow_cb:

        # Log parameters
        mlflow_cb.log_params({
            "experiment_id": experiment_id,
            "dataset": dataset,
            "exec_model": exec_model,
            "opt_model": opt_model,
            "max_rounds": max_rounds,
            "sample": sample,
            "validation_rounds": validation_rounds,
            "description": description,
        })

        # Log tags if provided
        if tags:
            mlflow.set_tags({f"tag_{i}": tag for i, tag in enumerate(tags)})

        # Attach callback to optimizer for logging round metrics
        optimizer_instance.mlflow_callback = mlflow_cb

        # Run optimization
        optimizer_instance.optimize("Graph")

        # Log final artifacts (workflows, results)
        workspace_path = Path(optimizer_instance.root_path) / "workflows"
        if workspace_path.exists():
            mlflow_cb.log_artifacts(str(workspace_path))
