#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for AFlow SLM research.
Reads experiment_config.yaml and runs experiments with MLflow tracking.
"""

import argparse
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig
from experiments.run_with_mlflow import run_experiment_with_mlflow


def load_config(config_path: str = "experiments/experiment_config.yaml"):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_llm_config(config_path: str = "config/config2.yaml"):
    """Load LLM configuration."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    # Extract models section
    if 'models' in config_data:
        config_data = config_data['models']
    return LLMsConfig(config_data)


def get_operators_for_dataset(dataset: str):
    """Get operator list based on dataset type."""
    # Based on CLAUDE.md guidelines
    if dataset in ["HumanEval", "MBPP"]:
        return ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
    elif dataset in ["GSM8K", "MATH"]:
        return ["Custom", "ScEnsemble", "Programmer"]
    elif dataset in ["HotpotQA", "DROP"]:
        return ["Custom", "AnswerGenerate", "ScEnsemble"]
    else:
        # Default
        return ["Custom", "ScEnsemble"]


def get_question_type(dataset: str):
    """Get question type based on dataset."""
    if dataset in ["HumanEval", "MBPP", "LiveCodeBench"]:
        return "code"
    elif dataset in ["GSM8K", "MATH"]:
        return "math"
    elif dataset in ["HotpotQA", "DROP"]:
        return "qa"
    else:
        return "qa"  # default


def run_single_experiment(exp_config: dict, defaults: dict, llm_config: LLMsConfig, dry_run: bool = False):
    """
    Run a single experiment.

    Args:
        exp_config: Experiment configuration dict
        defaults: Default parameters dict
        llm_config: LLM configuration
        dry_run: If True, only print what would be run
    """
    # Merge defaults with experiment-specific config
    dataset = exp_config["dataset"]
    exec_model = exp_config["exec_model"]
    opt_model = exp_config.get("opt_model", defaults.get("opt_model", "anthropic/claude-3.5-sonnet"))
    max_rounds = exp_config.get("max_rounds", defaults.get("max_rounds", 20))
    sample = exp_config.get("sample", defaults.get("sample", 4))
    validation_rounds = exp_config.get("validation_rounds", defaults.get("validation_rounds", 3))

    experiment_id = exp_config["id"]
    description = exp_config.get("description", "")
    tags = exp_config.get("tags", [])

    # Get operators and question type
    operators = get_operators_for_dataset(dataset)
    question_type = get_question_type(dataset)

    # Create unique workspace path with timestamp for this experiment
    # NOTE: Optimizer appends /{dataset} to optimized_path, so we structure as:
    # workspace_path (to Optimizer) = experiments/runs/{model_name_safe}/{experiment_id}_{timestamp}
    # Optimizer creates: experiments/runs/{model_name_safe}/{experiment_id}_{timestamp}/{dataset}/workflows/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use exact model name from config, sanitize for filesystem and Python imports
    # Replace "/", ".", and "-" to avoid Python import issues
    model_name_safe = exec_model.replace("/", "_").replace(".", "_").replace("-", "_")

    experiment_run_id = f"{experiment_id}_{timestamp}"
    workspace_path = f"experiments/runs/{model_name_safe}/{experiment_run_id}"
    final_path = f"{workspace_path}/{dataset}"  # What the final path will be after Optimizer appends dataset

    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_id}")
    print(f"Run ID: {experiment_run_id}")
    print(f"Description: {description}")
    print(f"Dataset: {dataset}")
    print(f"Exec Model: {exec_model}")
    print(f"Opt Model: {opt_model}")
    print(f"Max Rounds: {max_rounds}")
    print(f"Sample: {sample}")
    print(f"Validation Rounds: {validation_rounds}")
    print(f"Tags: {', '.join(tags)}")
    print(f"Workspace: {final_path}/workflows/")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Would execute experiment but skipping actual run.\n")
        return

    # Get LLM configs
    opt_llm_config = llm_config.get(opt_model)
    exec_llm_config = llm_config.get(exec_model)

    # Ensure workspace directory structure exists with __init__.py files
    # This is needed because AFlow imports workflows as Python modules
    # Create __init__.py at each level for Python module imports
    Path("experiments").mkdir(exist_ok=True)
    (Path("experiments") / "__init__.py").touch(exist_ok=True)

    Path("experiments/runs").mkdir(exist_ok=True)
    (Path("experiments/runs") / "__init__.py").touch(exist_ok=True)

    # Create model level (Optimizer will create dataset level inside this)
    model_path = Path(f"experiments/runs/{model_name_safe}")
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "__init__.py").touch(exist_ok=True)

    # Create experiment run level
    workspace_base = Path(workspace_path)
    workspace_base.mkdir(parents=True, exist_ok=True)
    (workspace_base / "__init__.py").touch(exist_ok=True)

    # The Optimizer will create the dataset directory, so we prepare paths for after that
    # Copy template and round_1 to workflows directory if they don't exist
    dataset_dir = Path(final_path)  # This is where Optimizer will create dataset subdirectory
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "__init__.py").touch(exist_ok=True)

    workflows_dir = dataset_dir / "workflows"
    workflows_dir.mkdir(exist_ok=True)
    (workflows_dir / "__init__.py").touch(exist_ok=True)

    # Copy template directory
    template_source = Path(f"workspace/{dataset}/workflows/template")
    template_dest = workflows_dir / "template"
    if template_source.exists() and not template_dest.exists():
        shutil.copytree(template_source, template_dest)
        print(f"Copied template from {template_source} to {template_dest}")

    # Copy round_1 directory (initial graph and prompt)
    round_1_source = Path(f"workspace/{dataset}/workflows/round_1")
    round_1_dest = workflows_dir / "round_1"
    if round_1_source.exists() and not round_1_dest.exists():
        shutil.copytree(round_1_source, round_1_dest,
                       ignore=shutil.ignore_patterns('*.csv', 'log.json', '__pycache__'))
        print(f"Copied round_1 from {round_1_source} to {round_1_dest}")

    # Create optimizer instance
    optimizer = Optimizer(
        dataset=dataset,
        question_type=question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=operators,
        sample=sample,
        check_convergence=False,  # Set to True if you want early stopping
        optimized_path=workspace_path,
        initial_round=1,
        max_rounds=max_rounds,
        validation_rounds=validation_rounds,
    )

    # Run with MLflow tracking
    run_experiment_with_mlflow(
        experiment_id=experiment_run_id,  # Use timestamped run ID
        dataset=dataset,
        exec_model=exec_model,
        opt_model=opt_model,
        max_rounds=max_rounds,
        sample=sample,
        validation_rounds=validation_rounds,
        optimizer_instance=optimizer,
        description=description,
        tags=tags,
    )

    print(f"\n✓ Completed experiment: {experiment_id}\n")


def main():
    parser = argparse.ArgumentParser(description="Run AFlow experiments from config file")
    parser.add_argument(
        "--config",
        default="experiments/experiment_config.yaml",
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--llm-config",
        default="config/config2.yaml",
        help="Path to LLM config YAML file"
    )
    parser.add_argument(
        "--experiment-id",
        help="Run only specific experiment by ID (e.g., 'qwen_trial2')"
    )
    parser.add_argument(
        "--tag",
        help="Run only experiments with specific tag (e.g., 'phase1')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running"
    )

    args = parser.parse_args()

    # Load configs
    exp_config = load_config(args.config)
    llm_config = load_llm_config(args.llm_config)

    experiments = exp_config.get("experiments", [])
    defaults = exp_config.get("defaults", {})

    # Filter experiments if needed
    if args.experiment_id:
        experiments = [e for e in experiments if e["id"] == args.experiment_id]
        if not experiments:
            print(f"Error: No experiment found with id '{args.experiment_id}'")
            sys.exit(1)

    if args.tag:
        experiments = [e for e in experiments if args.tag in e.get("tags", [])]
        if not experiments:
            print(f"Error: No experiments found with tag '{args.tag}'")
            sys.exit(1)

    print(f"\nFound {len(experiments)} experiment(s) to run\n")

    # Run experiments
    for exp in experiments:
        run_single_experiment(exp, defaults, llm_config, dry_run=args.dry_run)

    print("\n" + "="*80)
    print(f"All experiments completed! ({len(experiments)} total)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
