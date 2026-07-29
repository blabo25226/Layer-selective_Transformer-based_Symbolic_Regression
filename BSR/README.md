# Boolformer: symbolic regression of Boolean functions with transformers

This repository contains code for the paper '**Boolformer: symbolic regression of Boolean functions with transformers**'

## Installation

### Using UV (Recommended)

1. First, install UV:

```bash
pip install uv
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
uv venv

# Activate it (Unix/MacOS)
source .venv/bin/activate
# or on Windows
.venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

## Training and evaluation

### Training Models

To run the training of the noiseless model:

```bash
uv run python scripts/train.py -r <run_name> \
    -t config/transformer/noiseless.py \
    -f config/formula/noiseless.py \
    --nw <num_workers> \
    --bd <backup_directory> \
    --bs <batch_size> \
    --be <backup_every> \
    -d <device_id>
```

To run the training of the noisy model:

```bash
uv run python scripts/train.py -r <run_name> \
    -t config/transformer/noisy.py \
    -f config/formula/noisy.py \
    --nw <num_workers> \
    --bd <backup_directory> \
    --bs <batch_size> \
    --be <backup_every> \
    -d <device_id>
```

All hyperparameters not specified in the command arguments are contained in the configuration files, found inside `./config`.

Models are automatically saved during training in `.ckpt` format. To run the evaluation benchmark, the `.ckpt` file containing the model weights is needed.

### Run Predictions

To run prediction on synthetic data:

```bash
uv run python scripts/predict.py \
    --model <path_to_.ckpt> \
    -f <path_to_the_config> \
    --bs <batch_size> \
    --nw <num_workers> \
    -d <cuda_device_id> \
    -s <size> \
    --output <output_folder_path> \
    --beam <beam_size>
```

### PMLB Benchmark

To run the PMLB Benchmark:

```bash
uv run python scripts/benchmarks/pmlb/compute_results.py \
    --model <path_to_.ckpt> \
    --config config/formula/noisy.py \
    --beam <beam_size> \
    --nb_smpl <resampling_count> \
    --output <output_folder_path> \
    --device <cuda_device_id>
```

### GRN Benchmark

To run the GRN benchmark, please build the Docker image with `docker build -t grn .`. Spin up a container with `docker run -d --gpus all -it --name grnbench grn sleep infinity`. Attach to the container.

Inside the container, you will need to run `internal_run.py`, which is located in `/reviewAndAssessment/xsolver`. For the benchmark to work, it needs access to the model saved in `.ckpt` format.

The command:

```bash
python3 internal_run.py --device <inference_device> --model_path <path_to_.ckpt>
```

will generate the formulas for the GRN using Boolformer. Results are saved in `./results/Ecoli`.

Finally, run `gen_plots.py` to generate the radial plots.

### Generating Graphs

The repository includes several scripts for generating graphs to visualize results:

#### Noiseless Model Graphs

To generate graphs for noiseless model results:

```bash
uv run python scripts/graphs/noiseless_graphs.py \
    --input <path_to_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging> \
    --ft_size <font_size>
```

This will generate graphs showing the relationship between binary operators and metrics, formula distribution, and active variables impact.

#### Noisy Model Graphs

To generate graphs for noisy model results:

```bash
uv run python scripts/graphs/noisy_graphs.py \
    --input <path_to_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging>
```

This will create various graphs showing how different metrics (accuracy, F1 score, perfect recovery) are affected by factors like active variables, inactive variables, number of input points, and flip probability.

#### Temperature Analysis Graphs

To generate graphs analyzing the effect of temperature on model performance:

```bash
uv run python scripts/graphs/temperature_graphs.py \
    --input <path_to_temperature_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging>
```

This will create graphs showing how different metrics vary with temperature settings.

## Common Issues

If you encounter any UV-related warnings about hardlinking, you can suppress them by setting:

```bash
export UV_LINK_MODE=copy
```

or by using the `--link-mode=copy` option when running UV commands.
