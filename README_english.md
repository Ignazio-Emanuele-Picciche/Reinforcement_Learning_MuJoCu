# Reinforcement_Learning_MuJoCu

<p align="center">
  <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/halfcheetah.gif" alt="HalfCheetah GIF">
</p>

## Index
1. [Project Description](#project-description)
2. [Environment Setup](#environment-setup)
3. [Repository Structure](#repository-structure)
4. [Project Organization](#project-organization)
5. [Graphs and Results](#graphs-and-results)
6. [Challenges and Adopted Solutions](#challenges-and-adopted-solutions)

## Project Description
This project explores the application of advanced Reinforcement Learning (RL) algorithms for training and evaluating autonomous agents in the classic HalfCheetah and Ant environments from the Gymnasium library. The primary goal is to optimize the agents' performance using two main approaches for HalfCheetah: Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC). For the Ant environment, Twin Delayed Deep Deterministic Policy Gradient (TD3) is also employed.

The implementation leverages Optuna for automated hyperparameter search and includes customized environments to tailor the learning dynamics to the agents' specific requirements. The project also incorporates normalization techniques to stabilize training and evaluation tools for detailed performance analysis across different simulated scenarios.

## Environment Setup
Before running the code, ensure that you are using Python version 3.10.*. It is important to take some precautions and properly configure the environment. Follow these steps:

1. Create a Virtual Environment:

    - Open the terminal or command prompt.
    - Run the following command to create a virtual environment named "venv": `python -m venv venv`

2. Activate the Virtual Environment:

    - If you are using Windows: `./venv/Scripts/activate`
    - If you are using Unix or macOS: `source venv/bin/activate`

3. OPTIONAL - Deactivate the Virtual Environment (When Done):

    - Use the following command to deactivate the virtual environment: `deactivate`

4. Install Dependencies:

    - After cloning the project and activating the virtual environment, install the required dependencies using: `pip install -r requirements.txt`
    
    This command will download all the non-standard modules required by the application.

5. If the Python version used to create the virtual environment does not contain an updated version of pip, update pip using: `pip install --upgrade pip`

Once the virtual environment is set up and dependencies are installed, you are ready to run the application. Simply navigate to the desired `.ipynb` file and execute it.

## Repository Structure
The project is organized into the following directories and files:
```plaintext
REINFORCEMENT_LEARNING_MUJOCU/
├── Docs/                           # Documentation and results  
│   ├── Best Rewards.xlsx           # Excel file with the best results obtained  
│   ├── Explanation of RL Metrics.docx # Document explaining the metrics used  
│
├── Test Ant/                        # Experiments with the Ant environment  
│   ├── PPO/                          # Experiments with PPO  
│   │   ├── logs/                     # Training logs  
│   │   ├── ppo_Ant_tensorboard/      # TensorBoard files  
│   │   ├── ANT_train_PPO.ipynb       # Training notebook  
│   │   ├── HT_Ant_test_PPO.ipynb     # Hyperparameter tuning notebook  
│   │   ├── Render_ANT_PPO.ipynb      # Rendering notebook  
│   │   ├── ppo_Ant_model_PPO16.zip   # Saved model  
│   │   ├── ppo_Ant_model_PPO17_no_hp.zip  # Model without hyperparameter tuning  
│   │   ├── ppo_Ant_model_PPO18.zip   # Latest model version  
│   │   ├── vecnormalize_Ant.pkl      # Input normalization file  
│   │  
│   ├── SAC/                          # Experiments with SAC  
│   │   ├── logs/                     # Training logs  
│   │   ├── sac_Ant_tensorboard/      # TensorBoard files  
│   │   ├── ANT_train_SAC.ipynb       # Training notebook  
│   │   ├── HT_Ant_test_SAC.ipynb     # Hyperparameter tuning notebook    
│   │   ├── Render_ANT_SAC.ipynb      # Rendering notebook  
│   │   ├── sac_Ant_model.zip         # Saved model  
│   │   ├── vecnormalize_Ant.pkl      # Input normalization file  
│
... (rest of the repository structure remains unchanged)
```

## Project Organization
The project is structured into folders dedicated to testing two simulation environments: Ant and HalfCheetah. Within each of these folders, tests are further divided based on the reinforcement learning algorithm used for model training, including PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic), and TD3 (Twin Delayed Deep Deterministic Policy Gradient).

Each algorithm-specific folder contains the following Jupyter notebooks:

1. Hyperparameter Tuning: A dedicated notebook for tuning hyperparameters, essential for optimizing model performance before actual training.
2. Training: The main notebook for training the model in the respective environment, including summary graphs of training metrics for HalfCheetah.
3. Rendering: A notebook for visualizing and evaluating model behavior after training.

In the specific case of the PPO rendering notebook for the HalfCheetah environment, two rendering modes are available:

1. Human Mode: Displays real-time agent interaction with the environment, suitable for direct observation.
2. RGB Array Mode: Outputs rendering frames as RGB value arrays, allowing further processing and video creation for later analysis.

## Graphs and Results
To assess model performance, the project includes a comparison between the trained policy and a random policy, along with a graph showing the average reward progression during training.

1. Trained Policy vs. Random Policy
After training, the model is tested and compared with an agent that moves randomly (random policy). This comparison helps evaluate how much the algorithm has improved behavior compared to non-learning-based actions.

2. Average Reward Over Time
A graph tracking the model’s average reward during training provides insights into its improvement:
    - An increasing reward indicates that the model is learning to move better.
    - A stable or decreasing reward may suggest issues or a learning plateau.

Example graphs from PPO HalfCheetah:

<p align="center">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Reward.png" width="45%">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Valutazione_policy.png" width="45%">
</p>

## Challenges and Adopted Solutions
Initially, we used a basic predefined environment, which was inadequate for both the Cheetah and Ant models. Specifically, the Cheetah algorithm frequently converged to suboptimal solutions. To counter this, we introduced a penalty for excessive torso tilt, progressively increasing the penalty value to discourage undesirable behavior.

For Ant, various strategies were explored, fine-tuning key parameters to handle the environment's complexity. Despite the inherent challenges, we successfully identified and implemented the best solutions, significantly improving training outcomes while acknowledging room for further optimization.

