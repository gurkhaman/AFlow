# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AFlow is a framework for automatically generating and optimizing Agentic Workflows using Monte Carlo Tree Search (MCTS) in a code-represented workflow space. The system iteratively explores, refines, and evaluates workflows to find effective solutions for various tasks (code generation, math problems, QA).

## Development Setup

### Environment Setup
```bash
# Using uv (recommended)
uv venv --python 3.9
source .venv/bin/activate  # On Unix/macOS
# or `.venv\Scripts\activate` on Windows
uv pip install -r requirements.txt

# Or using conda
conda create -n aflow python=3.9
conda activate aflow
pip install -r requirements.txt
```

### Configuration
1. Copy `config/config2.example.yaml` to `config/config2.yaml`
2. Configure your LLM settings with API keys and base URLs:
```yaml
models:
  "<model_name>":
    api_type: "openai"
    base_url: "<your base url>"
    api_key: "<your api key>"
    temperature: 0
```

### Running the Optimizer
```bash
# Basic usage with a dataset
python run.py --dataset MATH

# With custom parameters
python run.py --dataset MATH --sample 4 --max_rounds 20 --optimized_path workspace

# Available datasets: HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP, LiveCodeBench
```

### Testing Mode
To test an existing workflow instead of optimizing:
- Edit `run.py` and change `optimizer.optimize("Graph")` to `optimizer.optimize("Test")`

## Architecture

### Core Concepts

**Code-Represented Workflows**: Workflows in AFlow are represented as Python code, where:
- **Nodes** are LLM invocations with parameters (model, prompt, temperature, output format)
- **Edges** are code structures (if-else, loops, function calls) that define execution flow
- This representation allows expressing any workflow pattern: sequential, parallel, conditional, or iterative

**Two-Model Architecture**: AFlow uses separate models for different purposes:
- **Optimizer Model** (default: Claude-3.5-sonnet): Generates and modifies workflows during MCTS search
- **Executor Model** (default: GPT-4o-mini): Runs the generated workflows on validation/test data
- This separation allows using stronger models for optimization and cheaper models for execution

### Core Components

**1. Node** (`scripts/` - various action node implementations)
- Basic unit of LLM invocation with configurable parameters:
  - `Model`: Which LLM to use
  - `Prompt`: Input/task description
  - `Temperature`: Randomness control (0-1)
  - `Output Format`: xml, json, markdown, or raw text
- See `metagpt_core/action_nodes/action_node.py` for the flexible interface

**2. Operator** (`scripts/operators.py`)
- Predefined combinations of Nodes that encode common agentic patterns
- Available operators:
  - `Custom`: Basic node construction with custom prompts
  - `AnswerGenerate`: Generate answers with structured thinking
  - `ScEnsemble`: Self-consistency ensemble (generate multiple solutions, vote)
  - `Programmer`: Generate and execute Python code for verification
  - `Test`: Execute code against test cases with reflection
  - `Review` & `Revise`: Iterative refinement pattern
  - `Format`: Structure output to specific format
- Custom operators can be added by extending the `Operator` base class
- Operators enhance search efficiency by providing known successful patterns

**3. Workflow** (`scripts/workflow.py`)
- Sequence of LLM-invoking nodes connected by edges
- Base class that must implement `__call__(self, problem: str)` method
- Generated workflows are stored in `workspace/<DATASET>/workflows/round_<N>/`
- Each workflow consists of:
  - `graph.py`: The workflow graph structure
  - `prompt.py`: Prompt templates used in the workflow

**4. Optimizer** (`scripts/optimizer.py`)
- Uses LLMs within a Monte Carlo Tree Search (MCTS) variant to discover optimal workflows
- **Key Innovation**: Each tree node represents a complete workflow (not individual LLM-invoking nodes)
- Two models are used:
  - `opt_model_name`: For optimization/search (default: claude-3-5-sonnet-20241022)
  - `exec_model_name`: For workflow execution (default: gpt-4o-mini)

**MCTS Process** (iterates for max 20 rounds or until convergence):

1. **Selection** - Choose parent workflow using soft mixed probability:
   - Formula: `P(i) = λ·(1/n) + (1-λ)·exp(α·(sᵢ-sₘₐₓ))/Σexp(α·(sⱼ-sₘₐₓ))`
   - Balances exploration (uniform) and exploitation (score-based)
   - λ=0.2, α=0.4 by default
   - Includes initial workflow to maintain persistent exploration

2. **Expansion** - LLM optimizer generates new workflow:
   - Receives parent workflow's tree-structured experience
   - Experience includes: all modifications, performance changes, execution logs
   - Makes single modification: add/modify/delete operators or prompts
   - Modifications stored in XML tags in optimizer's response

3. **Evaluation** - Execute new workflow on validation set:
   - Run 5 times for robustness (mean and std deviation)
   - Provides accurate feedback despite higher per-iteration cost
   - Validation set: 20% of data, selected for high score variance

4. **Backpropagation** - Update tree-structured experience:
   - Record: performance score, modification description, success/failure vs parent
   - Propagate experience back to parent workflow node
   - Add score to global record for future selection

**Convergence**: Stops early if top-k average score shows no improvement for n consecutive rounds (default: k=3, n=5)

- Utility modules in `scripts/optimizer_utils/`:
  - `graph_utils.py`: Workflow graph operations
  - `data_utils.py`: Data management
  - `experience_utils.py`: Experience tracking (tree structure)
  - `evaluation_utils.py`: Evaluation management
  - `convergence_utils.py`: Convergence checking

**5. Evaluator** (`scripts/evaluator.py`)
- Assesses workflow performance on given tasks
- Provides feedback to guide optimization towards more effective workflows
- Benchmark implementations in `benchmarks/`:
  - All benchmarks inherit from `BaseBenchmark` in `benchmarks/benchmark.py`
  - Implement: `evaluate_problem()`, `calculate_score()`, `get_result_columns()`

**6. LLM Integration** (`scripts/async_llm.py`)
- `AsyncLLM`: Async wrapper around OpenAI client for LLM calls
- `LLMConfig`: Configuration for individual LLM instances
- `LLMsConfig`: Manages multiple LLM configurations from YAML
- `ModelPricing`: Tracks cost information for different models

### Data Flow

1. **Initialization**:
   - Download datasets via `data/download_data.py` → stored in `data/datasets/`
   - Split: 20% validation (high variance samples), 80% test (fixed seed=42)
   - Start with blank template workflow W₀

2. **MCTS Iteration Loop** (for each round 1 to max_rounds):
   - **Select**: Choose parent workflow using soft mixed probability
   - **Expand**: LLM optimizer creates child workflow (single modification)
   - **Evaluate**: Execute child workflow 5x on validation set → average score
   - **Backpropagate**: Update tree experience with performance and modification
   - Track best workflow W* across all rounds

3. **Output**:
   - Generated workflows: `workspace/<DATASET>/workflows/round_<N>/graph.py`
   - Performance logs: `workspace/<DATASET>/workflows/round_<N>/{score}_{timestamp}.csv`
   - Tree-structured experience for each workflow node

4. **Testing** (optional):
   - Run best workflow W* on 80% test set
   - Report final metrics (solve rate, pass@1, F1, etc.)

## Adding Custom Datasets

To add a new dataset benchmark:

1. Create a new benchmark class in `benchmarks/your_dataset.py`:
```python
from benchmarks.benchmark import BaseBenchmark

class YourDatasetBenchmark(BaseBenchmark):
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        # Implement evaluation logic
        pass

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        # Implement scoring logic
        pass

    def get_result_columns(self) -> List[str]:
        # Return column names for results CSV
        return ["column1", "column2", "score", "cost"]
```

2. Add your dataset to `scripts/evaluator.py`:
```python
from benchmarks.your_dataset import YourDatasetBenchmark

DatasetType = Literal["HumanEval", "MBPP", ..., "YourDataset"]

# In Evaluator.__init__:
self.dataset_configs["YourDataset"] = YourDatasetBenchmark
```

3. Add dataset configuration to `run.py`:
```python
EXPERIMENT_CONFIGS["YourDataset"] = ExperimentConfig(
    dataset="YourDataset",
    question_type="qa",  # or "math", "code"
    operators=["Custom", "AnswerGenerate", "ScEnsemble"],
)
```

4. Add your dataset files to `data/datasets/`:
   - `yourdataset_validate.jsonl`: For validation during optimization
   - `yourdataset_test.jsonl`: For final testing

## Project Structure

```
AFlow/
├── run.py                    # Main entry point for optimization
├── run_baseline.py          # Baseline experiments
├── benchmarks/              # Dataset-specific evaluation logic
├── config/                  # Configuration files
├── data/                    # Dataset storage
│   ├── datasets/           # Downloaded datasets
│   └── download_data.py    # Dataset download utility
├── scripts/                # Core framework code
│   ├── optimizer.py        # MCTS-based workflow optimizer
│   ├── evaluator.py        # Benchmark evaluation coordinator
│   ├── workflow.py         # Base workflow class
│   ├── operators.py        # Operator implementations
│   ├── async_llm.py        # LLM client and configuration
│   ├── formatter.py        # Response formatting utilities
│   ├── optimizer_utils/    # Optimizer helper modules
│   ├── prompts/            # System prompts
│   └── utils/              # Common utilities
└── workspace/              # Generated workflows and results
    └── <DATASET>/
        └── workflows/
            ├── template/   # Operator templates
            └── round_<N>/  # Generated workflows per round
```

## Key Insights and Principles

### Why Code-Represented Edges?
- **Expressiveness**: Code can naturally express sequential, parallel, conditional, and iterative patterns
- **Flexibility**: No need for complex graph extensions (Petri nets, BPMN) to handle conditions/loops
- **Precision**: Standard programming constructs (if-else, for, while) provide precise control
- This is more comprehensive than graph-only (GPTSwarm) or network-only (earlier approaches) representations

### Tree-Structured Experience
- **Why it works**: Preserves complete optimization history without information loss
- **Key benefit**: Can accurately reuse successful modifications and avoid repeated failures
- **Structure**: Each tree node stores:
  - Parent workflow
  - Modification made
  - Performance score
  - Success/failure branches for child explorations
- **Contrast with ADAS**: ADAS stores workflows in a linear list, losing the relationship structure

### Execution Feedback
- **Direct performance measurement**: Run workflow 5 times, take average
- **Logs include**: Predictions, expected outputs, execution traces
- **Example impact**: On MATH dataset, helped LLM learn to format answers with `\boxed{}` LaTeX notation
- **Trade-off**: Higher cost per iteration, but leads to faster convergence overall

### Operators as Search Accelerators
- **Purpose**: Encode known successful patterns to guide search
- **Efficiency gain**: Significantly faster convergence with operators vs. without
- **Autonomous discovery**: Even without operators, AFlow can discover ensemble-like patterns
- **Flexibility**: Can work with empty operator set (just `Custom`) for full generality

### Dataset-Specific Operator Sets
Different domains benefit from different operator combinations:
- **Code** (HumanEval, MBPP): `Custom`, `CustomCodeGenerate`, `ScEnsemble`, `Test`
- **Math** (GSM8K, MATH): `Custom`, `ScEnsemble`, `Programmer`
- **QA** (HotpotQA, DROP): `Custom`, `AnswerGenerate`, `ScEnsemble`

## Important Notes

- The `workspace/` directory is gitignored and contains all generated workflows and experimental results
- Configuration file `config/config2.yaml` is gitignored - use the example file as a template
- Dataset files are large and downloaded separately via `download_data.py`
- Each optimization round generates complete executable workflow code in Python
- Workflows are represented as code (not just graphs) to allow flexible execution patterns
- The framework supports both synchronous and asynchronous LLM calls via `AsyncLLM`
- **Performance**: AFlow achieves 5.7% improvement over manually designed methods on average
- **Cost-effectiveness**: Can make weaker models outperform GPT-4o at 4.55% of the inference cost
- **Reproducibility**: Paper results available at Google Drive link (see `data/download_data.py`)

## Common Pitfalls

1. **Missing config file**: Must create `config/config2.yaml` from example before running
2. **Dataset not downloaded**: Run `download(["datasets"])` in `run.py` on first use
3. **Operator bugs**: Some operators may have bugs from MetaGPT migration (contact maintainers if issues arise)
4. **Validation set selection**: Can modify `va_list` in `evaluator.py` to use subset of validation data
5. **Model compatibility**: Different models may need different workflows - workflows are model-specific to some degree
