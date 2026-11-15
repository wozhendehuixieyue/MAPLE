# MAPLE: Mamba-based Prototype Learning for Sequential Recommendation

MAPLE is a high-performance sequential recommendation model built on the **Mamba** architecture. It enhances user representation by decoupling long-term and short-term interests and introducing **Global Interest Prototypes**. The model leverages Mamba's linear-time complexity for long-term sequence modeling and utilizes GRU for short-term dynamics, with a stable prototype update mechanism via Exponential Moving Average (EMA).

---

## Project Structure

Based on your directory layout, the recommended structure is as follows:

```text
MAPLE
├── dataset/             # Raw datasets and RecBole atomic files (.inter)
├── models/              # Model implementations
│   └── maple.py         # Core MAPLE model logic
├── saved/               # Saved model checkpoints (.pth)
├── config.yaml          # Hyperparameters and experimental settings
├── run.py               # Main entry point for training and evaluation
└── README.md            # Project documentation
```

---

## Key Features

1.  **Dual-Stream Encoder**:
    *   **Long-term Stream**: Powered by **Mamba** blocks to capture long-range dependencies with linear complexity.
    *   **Short-term Stream**: Powered by **GRU** to focus on local dynamics at the end of the sequence (controlled by `short_seq_len_ratio`).
2.  **Global Interest Prototypes**:
    *   Maintains long/short-term prototype matrices ($P_L$, $P_S$) representing latent global user intents.
    *   **EMA Update Mechanism**: Prototypes are updated via hard assignment and Exponential Moving Average (EMA) instead of backpropagation, ensuring stable cluster evolution.
3.  **Prototype Injection**:
    *   Extracts "Soft Proxies" using attention mechanisms to inject global intent information back into local user representations.
4.  **Cross-Intent Contrastive Learning**:
    *   A specialized contrastive loss that aligns long-term features with long-term proxies while pushing them away from short-term proxies to achieve effective decoupling.
5.  **Dynamic Gated Fusion**:
    *   A target-aware gating mechanism that adaptively balances the contributions of long-term and short-term representations.

---

## Configuration of the environment
The hardware and software we used are listed below to facilitate the environment's configuration. The detailed environment setting can be found in the `requirements.txt`. You can use pip install to reproduce the environment.

- Hardware:
  - CUDA: 12.8
- Software:
  - Python: 3.10
  - Pytorch: 2.7.0+cu128
- Usage
  - Install Causal Conv1d
    - `pip install causal-conv1d==1.5.4`
  - Install Recbole
    - `pip install recbole==1.2.1`
  - Install Mamba
    - `pip install mamba-ssm==2.2.6.post3` 

----

## Project Structure
You can download the raw dataset that needs to be preprocessed from https://github.com/RUCAIBox/RecSysDatasets, or download the preprocessed atomic files directly via the Baidu disk or Google Drive links provided by Recbole.

----

## Hyperparameters (`config.yaml`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `hidden_size` | int | Dimensionality of the hidden layers ($d_{model}$) |
| `mamba_d_state` | int | State dimension for Mamba blocks |
| `num_layers` | int | Number of Mamba layers |
| `short_seq_len_ratio`| float | Ratio of the sequence treated as "short-term" (0~1) |
| `num_proto_long` | int | Number of long-term prototypes ($K_L$) |
| `num_proto_short` | int | Number of short-term prototypes ($K_S$) |
| `ema_alpha` | float | Momentum coefficient for EMA prototype updates (default 0.99) |
| `tau_nce` | float | Temperature for InfoNCE contrastive loss |
| `lambda_align` | float | Weight for the contrastive learning loss |
| `injection_weight` | float | Weight of prototype information injected into features |

---

## Quick Start

### 1. Data Preparation
Place your RecBole-formatted dataset (e.g., `ml-1m`) into the `dataset/ml-1m/` folder.

### 2. Training the Model
Run the `run.py` script. Ensure the `MAPLE` class is imported correctly from the `models` directory. For example:

1. `python run.py --model SIGMA`
2. `python run.py --model MAPLE`
