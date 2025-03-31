# DRL_MMULS_Python

This is a Python implementation of the DRL_MMULS project, which focuses on reinforcement learning for inventory control in multi-item and single-item settings.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the single-item inventory control experiments:

```bash
python scripts/run_single_item.py
```

## Structure

- `src/ppo/`: Implementation of the PPO (Proximal Policy Optimization) algorithm
- `src/testbed/`: Environments for inventory control problems
  - `single_item/`: Single-item inventory system
  - `multi_item/`: Multi-item inventory system
- `scripts/`: Scripts to run experiments
- `data/`: Directory for storing experimental results

## Requirements

See requirements.txt for a list of dependencies. 