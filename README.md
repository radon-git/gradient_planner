# Sparse Reward Comparison Project

This project aims to compare the performance of three different reinforcement learning models in a sparse reward environment. The models being compared are:

1. **Custom Model**: A combination of a Dynamics Model and a Terminal Reward Model.
2. **PPO (Proximal Policy Optimization)**: A popular policy gradient method.
3. **SAC (Soft Actor-Critic)**: An off-policy actor-critic algorithm.

## Project Structure

The project is organized as follows:

- **notebooks/**: Contains Jupyter notebooks for experiments and analysis.
  - `performance_comparison.ipynb`: Implements the performance comparison experiment for the sparse reward environment.

- **src/**: Contains the source code for agents, models, and utilities.
  - **agents/**: Implementations of different agents.
    - `ppo_agent.py`: Contains the PPO agent implementation.
    - `sac_agent.py`: Contains the SAC agent implementation.
    - `planner_agent.py`: Contains the planner agent that uses the DynamicsModel and TerminalRewardModel.
  
  - **models/**: Defines the neural network architectures for the agents.
    - `ppo_models.py`: Defines the architecture for the PPO model.
    - `sac_models.py`: Defines the architecture for the SAC model.
    - `world_models.py`: Implements the DynamicsModel and TerminalRewardModel.
  
  - **utils/**: Contains utility functions and classes.
    - `environment_wrappers.py`: Implements the SparseRewardWrapper class.
    - `replay_buffers.py`: Implements replay buffer classes for experience storage.

- **saved_models/**: Directory to store the trained models.

- **data/**: Directory to store collected trajectory data from experiments.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sparse-reward-comparison
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook for performance comparison:
   ```
   jupyter notebook notebooks/performance_comparison.ipynb
   ```

## Results

The results of the performance comparison will be documented in the Jupyter notebook, including learning curves and evaluation metrics for each model. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details.