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
├── Docs/                           # Documentazione e risultati  
│   ├── Best Rewards.xlsx           # Foglio Excel con i miglior risultati ottenuti  
│   ├── Spiegazione metriche RL.docx # Documento con spiegazione delle metriche usate  
│
├── Test Ant/                        # Esperimenti con l'ambiente Ant  
│   ├── PPO/                          # Esperimenti con PPO  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── ppo_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_PPO.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_PPO.ipynb     # Notebook per il tuning dei parametri  
│   │   ├── Render_ANT_PPO.ipynb      # Notebook per il rendering del modello addestrato  
│   │   ├── ppo_Ant_model_PPO16.zip   # Modello salvato  
│   │   ├── ppo_Ant_model_PPO17_senza_hp.zip  # Modello senza hyperparameter tuning  
│   │   ├── ppo_Ant_model_PPO18.zip   # Ultima versione del modello  
│   │   ├── vecnormalize_Ant.pkl      # File per la normalizzazione degli input  
│   │  
│   ├── SAC/                          # Esperimenti con SAC  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── sac_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_SAC.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_SAC.ipynb     # Notebook per il tuning dei parametri    
│   │   ├── Render_ANT_SAC.ipynb      # Notebook per il rendering  
│   │   ├── sac_Ant_model.zip         # Modello salvato  
│   │   ├── vecnormalize_Ant.pkl      # File per la normalizzazione degli input  
│   │  
│   ├── TD3/                          # Esperimenti con TD3  
│   │   ├── td3_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_TD3.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_TD3.ipynb     # Notebook per il tuning dei parametri   
│   │   ├── Render_ANT_TD3.ipynb      # Notebook per il rendering  
│
├── Test Cheetah/                     # Esperimenti con l'ambiente HalfCheetah  
│   ├── PPO_CustomENV/                # Esperimenti con PPO  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── ppo_HalfCheetah_tensorboard/ # File di TensorBoard  
│   │   ├── HT_HalfCheetah_ppo.ipynb  # Notebook per il tuning dei parametri   
│   │   ├── ppo_cheetah.ipynb         # Notebook di addestramento  
│   │   ├── Render_HalfCheetah_ppo.ipynb # Notebook per il rendering  
│   │   ├── ppo_HalfCheetah_model.zip # Modello salvato  
│   │   ├── vecnormalize_HalfCheetah.pkl # File per la normalizzazione  
│   │  
│   ├── SAC_CustomENV/                # Esperimenti con SAC  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── sac_HalfCheetah_tensorboard/ # File di TensorBoard  
│   │   ├── HT_HalfCheetah_sac.ipynb  # Notebook per il tuning dei parametri   
│   │   ├── sac_cheetah.ipynb         # Notebook di addestramento  
│   │   ├── Render_HalfCheetah_sac.ipynb # Notebook per il rendering  
│   │   ├── sac_HalfCheetah_model.zip # Modello salvato  
│   │   ├── vecnormalize_HalfCheetah.pkl # File per la normalizzazione  
│
├── videos/                           # Cartella con video delle migliori policy  
│   ├── halfcheetah_best_policy.mp4   # Video della migliore policy  
│   ├── halfcheetah.gif               # GIF della simulazione  
│
├── venv/                             # Ambiente virtuale Python  
│
├── .gitignore                        # File per ignorare file non necessari su Git  
├── prova.ipynb                        # Notebook per il tuning dei parametri  
├── README.md                          # Questo file  
├── requirements.txt                   # File con le dipendenze del progetto  
```

## Project Organization
To assess how well the model has learned, the project includes a comparison between the trained policy and a random policy, along with a graph that shows the trend of the average reward during training.

1. Comparison Between Trained and Random Policy
After training, the model is tested and compared with an agent that moves randomly (random policy). This helps evaluate how much the algorithm has improved behavior compared to an action without learning. Ideally, the trained policy should exhibit smoother and more efficient movements compared to the random policy.

2. Average Reward Graph Over Time
During training, the average reward obtained by the model is recorded at time intervals. This graph helps understand how the model is improving over time:

    - If the reward increases, it means the model is learning to move better.
    - If the reward stabilizes or decreases, it could indicate a problem or that the model has reached its maximum learning potential.

These tools help evaluate the quality of training and compare the different algorithms used.

The project is structured into folders dedicated to testing two simulation environments: Ant and HalfCheetah. Within each of these folders, tests are further divided based on the reinforcement learning algorithm used for model training, including PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic), and TD3 (Twin Delayed Deep Deterministic Policy Gradient).

Each algorithm-specific folder contains the following Jupyter notebooks:

1. Hyperparameter Tuning: A dedicated notebook for tuning hyperparameters, essential for optimizing model performance before actual training.

2. Training: The main notebook for training the model in the respective environment, which in the case of HalfCheetah also includes the visualization of summary graphs of training metrics.

3. Rendering: A notebook for visualizing and evaluating the model's behavior after training.

In the specific case of the PPO rendering notebook for the HalfCheetah environment, two rendering modes are available:

1. Human Mode: This mode allows real-time visualization of the agent's interaction with the environment, making it suitable for direct user observation.
2. RGB Array Mode: This mode returns rendering frames as arrays of RGB values, enabling further processing and video saving for later analysis or animation creation.

This organization enables a clear and modular management of experiments, facilitating the analysis of the performance of different algorithms applied to simulation environments.

## Graphs and Results
To assess how well the model has learned, the project includes a comparison between the trained policy and a random policy, along with a graph that shows the trend of the average reward during training.

1. Comparison Between Trained and Random Policy
After training, the model is tested and compared with an agent that moves randomly (random policy). This helps evaluate how much the algorithm has improved behavior compared to an action without learning. Ideally, the trained policy should exhibit smoother and more efficient movements compared to the random policy.

2. Average Reward Graph Over Time
During training, the average reward obtained by the model is recorded at time intervals. This graph helps understand how the model is improving over time:

    - If the reward increases, it means the model is learning to move better.
    - If the reward stabilizes or decreases, it could indicate a problem or that the model has reached its maximum learning potential.

These tools help evaluate the quality of training and compare the different algorithms used.

Example graphs from PPO HalfCheetah:

<p align="center">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Reward.png" width="45%">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Valutazione_policy.png" width="45%">
</p>

## Challenges and Adopted Solutions
At the beginning of the development, we used a predefined basic environment for the libraries, but we found that it was not suitable for either the Cheetah or Ant models. Specifically, for Cheetah, the algorithm often tended to converge towards suboptimal solutions rather than finding the global optimum. To improve performance, we introduced an additional penalty whenever the Cheetah's torso exceeded a certain angle, progressively increasing the penalty value to discourage such undesirable behaviors.

A further improvement was achieved by modifying the neural network architecture through the parameter "policy_kwargs": dict(net_arch=[256, 256, 128]), which led to more satisfactory results in the case of Cheetah.

Regarding Ant, we explored various strategies to improve performance, adapting and testing different algorithms with the goal of optimizing model control. In particular, we refined the neural network architecture and adjusted key parameters to handle the greater computational complexity of the environment. Despite the intrinsic difficulties of the task, we managed to identify and implement the best solutions among those tested, thus significantly advancing training and achieving concrete results, although there is still room for improvement to reach a definitive optimal solution.

