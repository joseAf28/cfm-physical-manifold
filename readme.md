# Physics Consistent Conditional Flow Matching Model as a Plasma Physics Surrogate Model

Code repository for the paper: [*Physics Consistent Conditional Flow Matching Model as a Plasma Physics Surrogate Model*]().

## Abstract
> Physics-informed surrogate modeling for steady-state systems often struggles with data scarcity and optimization instabilities from soft penalty losses. We introduce a unified, continuous-time Conditional Flow Matching (CFM) framework for surrogate modeling and sparse data assimilation, evaluated on low-temperature plasma dynamics. By combining data-dependent stochastic interpolants with dynamic masking, the framework unifies unconditioned prediction and progressive missing data imputation into a single objective. We replace standard velocity matching ($v$-prediction) with a direct state regression ($y$-prediction) objective, eliminating the variance-induced statistical inefficiencies of deterministic physics. For inference, the formulation supports both standard time-dependent ODE integration (CFM-$t$) and an autonomous, clock-free regime (CFM-$0$) that enables exact measurement conditioning via anchor overwriting without boundary disruption. Global conservation laws are enforced via post-hoc hard geometric projections rather than loss regularizers. Results indicate that both CFM models achieve superior accuracy, sample efficiency under data scarcity, and raw physical consistency compared to standard MLPs. Furthermore, macroscopic constraints alone yield negligible reductions in $L_2$ error, confirming the necessity of a data-driven generative manifold prior.

---


## Repository Structure

*   `configs/`: Contains the `.yaml` configuration files to set model parameters, training hyperparameters, and data paths.
*   `data/`: Contains the experimental dataset from  [Physics Consistent ML Github](https://github.com/matildevalente/physics_consistent_machine_learning/tree/main/data/ltp_system).
*   `src/`: Contains the core source code:
    *   `dataset.py`: Data loading, scaling, and PyTorch dataset wrappers.
    *   `models.py`: Architecture definitions (CFM_t, CFM_0, Regressor).
    *   `train.py`: Training loops for the generative and baseline models.
    *   `inference.py`: Generalized ODE solvers and flow matching functions.
    *   `losses.py`: Target interpolants and loss functions.
    *   `physics_constraints.py`: PyTorch differentiable constraints and CasADi hard projection solvers.
    *   `experiment_evaluator.py`: Evaluation pipelines for metrics (RMSE) and uncertainty quantification (UQ).

## Setup & Installation

1. Clone the repository.
2. Install the required dependencies. Ensure you have PyTorch, Scikit-Learn, and CasADi installed:
   ```bash
   pip install torch torchvision numpy scikit-learn casadi