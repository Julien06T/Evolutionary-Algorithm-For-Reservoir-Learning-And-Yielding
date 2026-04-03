# Evolutionary-Algorithm-For-Reservoir-Learning-And-Yielding

This repository implements an evolutionary algorithm for optimizing reservoir computing networks. The framework supports supervised learning tasks, such as chaotic time series forecasting, using modular components for mutation, selection, and evaluation.

## Features
- Evolutionary optimization of reservoir size, connectivity, and hyperparameters.
- Support for feedback connections and cycle detection.
- Parallel evaluation using `ray` and `joblib`.
- Task-specific fitness evaluation via `stream_dataset`.

## Requirements
- Python >= 3.8
- Dependencies listed in `requirements.txt` 

## Usage
1. Set task parameters in `main.py` (e.g., `task_name`, `generations`, `pop_size`).
2. Run `python main.py` to start evolution.
3. Results are printed to stdout; the best individual is saved and visualized.

## Directory Structure
- `main.py`: Entry point and evolutionary loop.
- `utils.py`: Core evolutionary operations.
- `requirements.txt`: List of dependencies.

## Dependencies
- `reservoirpy`: Reservoir computing backend.
- `scikit-learn`, `numpy`: Machine learning and numerical operations.
- `graphviz`: Visualization of the best individual.
- `stream_dataset`: Generate the tasks

