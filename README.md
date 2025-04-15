# DA-MBRL: Data-Assimilated Model-Based Reinforcement Learning

In this project, we tackle control from partial noisy observations.  
We propose a framework, **Data-assimilated Model-based Reinforcement Learning (DA-MBRL)**, that has three components:

1. &nbsp;&nbsp;*A predictive model of the system's dynamics* (physical or data-driven)
2. &nbsp;&nbsp;*An ensemble-based data assimilation method for real-time state estimation* (Ensemble Kalman Filter, EnKF)  
3. &nbsp;&nbsp;*An off-policy actor-critic reinforcement learning (RL) algorithm for learning the control policy* (Deep Deterministic Policy Gradient, DDPG)

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f6c3dfb-740d-4b98-8468-1e94584ed8d9" alt="da_mbrl_diagram" width="500"/>
</p>

## Environment
The environment is the **Kuramoto–Sivashinsky (KS) equation**, which is a 1D partial differential equation that exhibits *spatiotemporal chaos*. 
The below figure shows the KS system first evolving without control and then being controlled using a trained RL agent, which stabilises the flow. We place sensors along the x-axis to obtain measurements of the system. We aim to learn a stabilising optimal control policy using these partial and noisy measurements, i.e., observations. The control is enabled by actuators applying a Gaussian mixture forcing.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d5276661-7765-4bc5-b843-ecee38a6fe50" alt="ks_illustration" width="1000"/>
</p>

## Running Experiments
The codebase is written in **[JAX](https://github.com/google/jax)**.
The following experiments can be run:

1. **Model-free RL**  
   ```bash
   python ddpg_experiment_v3.py
   ```
2. **Data-assimilated model-based RL** using a physical *truncated Fourier basis* model of the system
   ```bash
   python ddpg_with_enkf_experiment_v3.py
   ```
4. **Data-assimilated model-based RL** using a data-driven *control-aware Echo State Network (ESN)* model of the system
   ```bash
   python ddpg_with_enkf_esn.py
   ```
Running an experiment creates a folder in `local_results/` where configurations, model weights and plots (optional) are saved. The results are visualised in the Jupyter notebooks, [`Model-free`](./mf_results.ipynb), [`Model-based Fourier`](./fourier_results.ipynb) and [`Model-based ESN`](./esn_results.ipynb). 

### Configuration

The experiments can be configured using [`ml_collections`](https://github.com/google/ml_collections). You can find sample config files in the [`configs/`](./configs) directory.

To specify a configuration file when running an experiment, use the `--config` flags, or you can also individually configure the setting. For example:

```bash
python ddpg_with_enkf_esn.py --config configs/enKF_config_mb.py --config.enKF.std_obs 0.1 --env_config configs/KS_config.py --env_config.nu 0.08
```
#### Optional flags:
--make_plots – generate plots of the episodes during training


--log_wandb – log losses and metrics to [`Weights and Biases`](https://wandb.ai)
Previous runs can be accessed at:
* [`Model-free`](https://wandb.ai/defneozan/DA-RL-MF)
* [`Model-based Fourier`](https://wandb.ai/defneozan/DA-RL-MB-Fo)
* [`Model-based ESN`](https://wandb.ai/defneozan/DA-RL-MB-ESN)
