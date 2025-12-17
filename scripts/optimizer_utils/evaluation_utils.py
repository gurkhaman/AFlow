from scripts.evaluator import Evaluator


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # Load graph with graph_utils from optimizer
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        return sum_score / validation_n

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
        )

    async def evaluate_graph_test_robust(self, optimizer, directory, test_runs=1):
        """
        Run test evaluation multiple times for robustness.

        Args:
            optimizer: Optimizer instance
            directory: Round directory path
            test_runs: Number of test runs (default: 1)

        Returns:
            tuple: (avg_score, scores_list, avg_cost, total_cost)
        """
        from scripts.evaluator import Evaluator
        from scripts.logs import logger

        evaluator = Evaluator(eval_path=directory)
        scores = []
        total_cost_sum = 0

        for i in range(test_runs):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=True,
            )
            scores.append(score)
            total_cost_sum += total_cost

            # Log individual test run to MLflow if callback exists
            if optimizer.mlflow_callback:
                optimizer.mlflow_callback.log_metric("test_score", score, step=i+1)

        avg_score = sum(scores) / len(scores)
        avg_cost = total_cost_sum / len(scores) if len(scores) > 0 else 0

        return avg_score, scores, avg_cost, total_cost_sum


async def run_test_on_best_round(
    dataset: str,
    root_path: str,
    exec_llm_config,
    mlflow_callback=None,
    test_runs: int = 1
) -> dict:
    """
    Standalone test evaluation on best validation round.

    Can be called from Optimizer or directly from manual scripts.

    Args:
        dataset: Dataset name (e.g., "HumanEval", "GSM8K")
        root_path: Path to experiment (e.g., "experiments/runs/.../HumanEval")
        exec_llm_config: LLM configuration for workflow execution
        mlflow_callback: Optional MLflow callback for logging
        test_runs: Number of test runs (default: 1)

    Returns:
        dict with: score, scores, avg_cost, total_cost, best_round, best_val_score, results_path
    """
    import os
    from pathlib import Path
    from scripts.optimizer_utils.data_utils import DataUtils
    from scripts.optimizer_utils.graph_utils import GraphUtils
    from scripts.logs import logger

    # Convert absolute path to relative path from cwd (required for load_graph module import)
    root_path = str(root_path)
    if os.path.isabs(root_path):
        try:
            root_path = os.path.relpath(root_path, os.getcwd())
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            pass

    # Initialize utilities
    data_utils = DataUtils(root_path)
    graph_utils = GraphUtils(root_path)

    graph_path = f"{root_path}/workflows"

    logger.info("=" * 80)
    logger.info("Starting test evaluation on best validation round...")
    logger.info("=" * 80)

    # Get best validation round
    try:
        best_round_info = data_utils.get_best_validation_round()
        best_round = best_round_info["round"]
        best_val_score = best_round_info["score"]
        logger.info(f"Best validation round: {best_round} (score: {best_val_score:.5f})")
    except Exception as e:
        logger.error(f"Failed to identify best validation round: {e}")
        raise

    # Load best workflow
    directory = f"{graph_path}/round_{best_round}"

    try:
        graph = graph_utils.load_graph(best_round, graph_path)
        logger.info(f"Loaded workflow from round {best_round}")
    except Exception as e:
        logger.error(f"Failed to load workflow for round {best_round}: {e}")
        raise

    # Run test evaluation
    logger.info(f"Running test evaluation {test_runs} time(s)...")

    evaluator = Evaluator(eval_path=directory)
    scores = []
    total_cost_sum = 0

    for i in range(test_runs):
        score, avg_cost, total_cost = await evaluator.graph_evaluate(
            dataset,
            graph,
            {"dataset": dataset, "llm_config": exec_llm_config},
            directory,
            is_test=True,
        )
        scores.append(score)
        total_cost_sum += total_cost

        # Log individual test run to MLflow if callback exists
        if mlflow_callback:
            mlflow_callback.log_metric("test_score", score, step=i + 1)

    avg_score = sum(scores) / len(scores)
    avg_cost_per_run = total_cost_sum / len(scores) if len(scores) > 0 else 0

    logger.info(f"Test evaluation completed!")
    logger.info(f"  Individual scores: {[f'{s:.5f}' for s in scores]}")
    logger.info(f"  Average score: {avg_score:.5f}")
    logger.info(f"  Total cost: ${total_cost_sum:.5f}")

    # Save test results to results.json
    data = data_utils.load_results(graph_path)
    for score in scores:
        run_cost = total_cost_sum / len(scores)
        new_data = data_utils.create_result_data(
            best_round, score, avg_cost_per_run, run_cost, is_test=True
        )
        data.append(new_data)

    result_path = data_utils.get_results_file_path(graph_path)
    data_utils.save_results(result_path, data)

    # Log to MLflow
    if mlflow_callback:
        mlflow_callback.log_metric("test_score_avg", avg_score)
        mlflow_callback.log_metric("test_cost_total", total_cost_sum)
        logger.info("Test metrics logged to MLflow")

    logger.info(f"Results saved to {result_path}")
    logger.info("=" * 80)
    logger.info("Test evaluation complete!")
    logger.info("=" * 80)

    return {
        "score": avg_score,
        "scores": scores,
        "avg_cost": avg_cost_per_run,
        "total_cost": total_cost_sum,
        "best_round": best_round,
        "best_val_score": best_val_score,
        "results_path": result_path,
    }
