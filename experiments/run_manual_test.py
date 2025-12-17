#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manually trigger test evaluation on the best validation round.

Usage:
    python experiments/run_manual_test.py experiments/runs/microsoft_Phi_3_mini_4k_instruct/phi3_humaneval_trial2_20251217_110507/HumanEval

    # Disable MLflow logging
    python experiments/run_manual_test.py experiments/runs/.../HumanEval --no-mlflow

    # Specify executor model explicitly
    python experiments/run_manual_test.py experiments/runs/.../HumanEval --executor-model "microsoft/Phi-3-mini-4k-instruct"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.optimizer_utils.evaluation_utils import run_test_on_best_round
from scripts.async_llm import LLMsConfig
from scripts.logs import logger


def infer_executor_model(experiment_path: Path) -> str:
    """Infer executor model from experiment path."""
    for part in experiment_path.parts:
        if "Qwen" in part and "3B" in part:
            return "Qwen/Qwen2.5-3B-Instruct"
        elif "Phi" in part and "3" in part:
            return "microsoft/Phi-3-mini-4k-instruct"
        elif "Llama" in part and "3B" in part:
            return "meta-llama/Llama-3.2-3B-Instruct"
    return "gpt-4o-mini"


async def main_async(args):
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    # Dataset is the folder name
    dataset = experiment_path.name

    # Get executor model
    executor_model = args.executor_model or infer_executor_model(experiment_path)

    # Load LLM config
    llms_config = LLMsConfig.default()
    exec_llm_config = llms_config.get(executor_model)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Executor model: {executor_model}")

    # Setup MLflow if enabled
    mlflow_callback = None
    if not args.no_mlflow:
        try:
            from experiments.run_with_mlflow import MLflowCallback

            experiment_id = experiment_path.parent.name
            run_name = f"{experiment_id}_manual_test"

            mlflow_callback = MLflowCallback(
                experiment_name=f"aflow_{dataset.lower()}",
                run_name=run_name
            )
            mlflow_callback.__enter__()

            # Log params
            mlflow_callback.log_params({
                "experiment_id": experiment_id,
                "dataset": dataset,
                "exec_model": executor_model,
                "manual_test": True,
            })
        except ImportError:
            logger.warning("MLflow not available, continuing without logging")

    try:
        result = await run_test_on_best_round(
            dataset=dataset,
            root_path=str(experiment_path),
            exec_llm_config=exec_llm_config,
            mlflow_callback=mlflow_callback,
            test_runs=1
        )
        return result
    finally:
        if mlflow_callback:
            mlflow_callback.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Run test evaluation on best validation round",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "experiment_path",
        help="Path to experiment (e.g., experiments/runs/.../HumanEval)"
    )
    parser.add_argument(
        "--executor-model",
        default=None,
        help="Executor model name (inferred from path if not provided)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_async(args))
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
