# Reinforcement_Learning_MuJoCu

<p align="center">
    <img src="../Test Cheetah/media/halfcheetah.gif" alt="HalfCheetah GIF" width="40%">
    <img src="ANT_media/rendering_ant_ppo15.gif" alt="Ant GIF" width="40%">
</p>

To read the README.md in Italian, click [here](../README.md).

## Index 
1. [Project Description](#1-project-description)
2. [Environment Setup](#2-environment-setup)
3. [Repository Structure](#3-repository-structure)
4. [Project Organization](#4-project-organization)
    - [Folder Structure](#41-folder-structure)
    - [Rendering Modes](#42-rendering-modes)
    - [Organization Advantages](#43-organization-advantages)

5. [Graphs and Results](#5-graphs-and-results)
    - [Comparison between Trained and Random Policy](#51-comparison-between-trained-and-random-policy)
    - [Average Reward Graph over Time](#52-average-reward-graph-over-time)
    - [Graphs](#53-graphs)

6. [Challenges and Solutions](#6-challenges-and-solutions)
    - [Challenges and Solutions with the Cheetah Model](#61-challenges-with-the-cheetah-model)
    - [Challenges and Solutions with the Ant Model](#62-challenges-with-the-ant-model)

## 1. Project Description
This project explores the application of advanced Reinforcement Learning (RL) algorithms for training and evaluating autonomous agents in the classic HalfCheetah and Ant environments from the Gymnasium library. The main objective is to optimize the agents' performance using three main approaches:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**

The implementation leverages Optuna for automated hyperparameter search, optimizing configurations to maximize agent performance. Additionally, it includes custom environments to adapt learning dynamics to the specific needs of the agents, improving training robustness and effectiveness.

The project employs normalization techniques to stabilize training, reducing observation variance and improving algorithm convergence. Detailed evaluation tools, such as average reward graphs and comparisons between trained and random policies, are used to analyze agent performance in various simulated scenarios. These tools allow for quick identification of improvement areas and evaluation of the effectiveness of different RL strategies implemented.

## 2. Environment Setup
Before running the code, make sure to use Python version 3.10.*. It is important to take some precautions and properly configure the environment. Follow these steps:

1. Create a Virtual Environment:

    - Open the terminal or command prompt.
    - Run the following command to create a virtual environment called "venv": `python -m venv venv`

2. Activate the Virtual Environment:

    - If you are using Windows: `.\venv\Scripts\activate`
    - If you are using Unix or macOS: `source venv/bin/activate`

3. OPTIONAL - Deactivate the Virtual Environment (When you are done):

    - Use the following command to deactivate the virtual environment: `deactivate`

4. Install Dependencies:

    - After cloning the project and activating the virtual environment, install the required dependencies using: `pip install -r requirements.txt`
    - This command will download all the non-standard modules required by the application.

5. If the Python version used to create the virtual environment does not contain an updated version of pip, update pip using: `pip install --upgrade pip`

Once the virtual environment is configured and dependencies are installed, you are ready to run the application. Simply navigate to the desired .ipynb file and execute it.

## 3. Repository Structure
The repository is organized to facilitate navigation and management of files and folders. The main structure of the project is as follows:

```plaintext
REINFORCEMENT_LEARNING_MUJOCU/
├── Docs/                           # Documentation and results  
│   ├── Best Rewards.xlsx           # Excel sheet with the best results obtained  
│   ├── Spiegazione metriche RL.docx # Document explaining the metrics used  
│
├── Test Ant/                        # Experiments with the Ant environment  
│   ├── PPO/                          # Experiments with PPO  
│   │   ├── logs/                     # Training logs  
│   │   ├── ppo_Ant_tensorboard/      # TensorBoard files  
│   │   ├── ANT_train_PPO.ipynb       # Training notebook  
│   │   ├── HT_Ant_test_PPO.ipynb     # Hyperparameter tuning notebook  
│   │   ├── Render_ANT_PPO.ipynb      # Rendering notebook  
│   │   ├── ppo_Ant_model_PPO16.zip   # Saved model  
│   │   ├── ppo_Ant_model_PPO17_senza_hp.zip  # Model without hyperparameter tuning  
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
│   │  
│   ├── TD3/                          # Experiments with TD3  
│   │   ├── td3_Ant_tensorboard/      # TensorBoard files  
│   │   ├── ANT_train_TD3.ipynb       # Training notebook  
│   │   ├── HT_Ant_test_TD3.ipynb     # Hyperparameter tuning notebook   
│   │   ├── Render_ANT_TD3.ipynb      # Rendering notebook  
│
├── Test Cheetah/                     # Experiments with the HalfCheetah environment  
│   ├── PPO_CustomENV/                # Experiments with PPO  
│   │   ├── logs/                     # Training logs  
│   │   ├── ppo_HalfCheetah_tensorboard/ # TensorBoard files  
│   │   ├── HT_HalfCheetah_ppo.ipynb  # Hyperparameter tuning notebook   
│   │   ├── ppo_cheetah.ipynb         # Training notebook  
│   │   ├── Render_HalfCheetah_ppo.ipynb # Rendering notebook  
│   │   ├── ppo_HalfCheetah_model.zip # Saved model  
│   │   ├── vecnormalize_HalfCheetah.pkl # Input normalization file  
│   │  
│   ├── SAC_CustomENV/                # Experiments with SAC  
│   │   ├── logs/                     # Training logs  
│   │   ├── sac_HalfCheetah_tensorboard/ # TensorBoard files  
│   │   ├── HT_HalfCheetah_sac.ipynb  # Hyperparameter tuning notebook   
│   │   ├── sac_cheetah.ipynb         # Training notebook  
│   │   ├── Render_HalfCheetah_sac.ipynb # Rendering notebook  
│   │   ├── sac_HalfCheetah_model.zip # Saved model  
│   │   ├── vecnormalize_HalfCheetah.pkl # Input normalization file  
│
├── media/                           # Folder with videos/photos of the best policies  
│   ├── halfcheetah_best_policy.mp4   # Video of the best policy  
│   ├── halfcheetah.gif               # Simulation GIF  
│
├── venv/                             # Python virtual environment  
│
├── .gitignore                        # File to ignore unnecessary files on Git  
├── README.md                          # This file  
├── requirements.txt                   # Project dependencies file  
```

## 4. Project Organization 

The project is structured to facilitate the management and analysis of reinforcement learning experiments on two simulation environments: Ant and HalfCheetah. Each environment has its own dedicated folder, within which tests are further divided based on the reinforcement learning algorithm used for model training. The main algorithms used are:

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)

### 4.1 Folder Structure

Each folder dedicated to an algorithm contains the following Jupyter notebooks, each with a specific purpose:

1. **Hyperparameter Tuning**: This notebook is dedicated to hyperparameter tuning. Optimizing hyperparameters is crucial to improve model performance before the actual training phase. We use tools like Optuna to automate this process and find the best possible configurations.

2. **Training**: The main notebook for training the model in the corresponding environment. In this notebook, the model is trained using the specified algorithm. For the HalfCheetah environment, the notebook also includes summary graphs of training metrics, such as average reward over time, to monitor model progress.

3. **Rendering**: A notebook dedicated to visualizing and evaluating the model's behavior after training. This notebook allows observing how the agent interacts with the environment and qualitatively evaluating its performance.

### 4.2 Rendering Modes

In the specific case of the PPO model rendering notebook for the HalfCheetah environment, two rendering modes are available:

1. **Human Mode**: This mode allows real-time visualization of the agent's interaction with the environment. It is particularly useful for direct observation by the user, allowing immediate viewing of the agent's behavior.

2. **RGB Array Mode**: This mode returns the rendering frames as an array of RGB values. This format is useful for processing and saving the video for later analysis or creating animations. It can be used to generate GIFs or videos showing the evolution of the agent's behavior over time.

### 4.3 Organization Advantages

This modular organization allows for clear and structured management of experiments. Each phase of the reinforcement learning process, from hyperparameter tuning to training and evaluation, is separated into distinct notebooks. This approach facilitates the analysis of the performance of different algorithms applied to the simulation environments and allows for quick identification of improvement areas.

Additionally, the division into specific folders and notebooks makes the project easily navigable and understandable, even for those unfamiliar with the code. Each notebook is documented and contains detailed explanations of the steps performed, making the project accessible and useful for educational and research purposes.

## 5. Graphs and Results
To evaluate the effectiveness of the model training, the project includes a comparison between the trained policy and a random policy, as well as graphs showing the trend of the average reward during training.

### 5.1 Comparison between Trained and Random Policy
After training, the model is tested and compared with an agent that moves randomly (random policy). This comparison serves to highlight how much the algorithm has improved behavior compared to an action without learning. Ideally, the trained policy should show smoother and more efficient movements compared to the random policy, demonstrating the effectiveness of the reinforcement learning algorithm.

### 5.2 Average Reward Graph over Time
During training, the average reward obtained by the model is recorded at time intervals. This graph is essential for monitoring the model's progress and identifying any issues:

- **Reward Increase**: If the average reward increases over time, it means that the model is learning to move better and optimize its actions.
- **Reward Stabilization or Decrease**: If the average reward stabilizes or decreases, it could indicate that the model has reached its maximum learning level or that there are training issues that need to be addressed.

### 5.2 Graphs
Below are some of the best results obtained during training with the PPO algorithm for the HalfCheetah environment:

- **Average Reward Graph**: Shows the trend of the average reward over time, highlighting the model's progress during training.
- **Policy Evaluation**: Compares the performance of the trained policy with that of a random policy, demonstrating the algorithm's effectiveness.

<p align="center">
    <img src="Test Cheetah/media/Reward.png" width="45%">
    <img src="Test Cheetah/media/Valutazione_policy.png" width="45%">
</p>


Below are some of the best results obtained during training with the PPO algorithm for the Ant environment:

<p align="left">
    <img src="myLib/ANT_media/best_eval_ANT_PPO.png" width="70%">
    <img src="myLib/ANT_media/best_train_ANT_PPO.png" width="70%">
</p>

These evaluation tools are essential for understanding the quality of training and comparing the performance of the different algorithms used in the project.


## 6. Challenges and Solutions

During the early stages of development, we used a predefined base environment for the libraries. However, we soon realized that this environment was not adequate for either the Cheetah model or the Ant model. 

### 6.1 Challenges with the Cheetah Model

For the Cheetah model, the algorithm often tended to converge to suboptimal solutions rather than finding the global optimum. This behavior was due to a lack of adequate penalties for certain undesirable movements. To improve performance, we introduced an additional penalty if the Cheetah's back exceeded a certain angle. By progressively increasing the penalty value, we were able to discourage such undesirable behaviors, leading the algorithm to converge to more optimal solutions.

#### Custom Penalty

We implemented a custom wrapper for the `HalfCheetah-v5` environment that modifies the reward function. This wrapper penalizes the robot if the torso tilts too far back (indicating a fall), increasing the penalty over time as it remains in this position.

- **CustomRewardWrapper**: This wrapper modifies the environment's reward by penalizing the robot if the torso tilts beyond a certain threshold. The penalty increases based on the time spent in this condition, discouraging undesirable behaviors.

#### Environment Creation

To create the environment with the custom reward, we used a function that configures the `HalfCheetah-v5` environment with optimized parameters and applies the `CustomRewardWrapper`.

- **make_env**: This function creates and returns an instance of the `HalfCheetah-v5` environment with custom parameters, performance monitoring, and application of the `CustomRewardWrapper`.

#### Neural Network Architecture Modification

Additionally, we achieved further improvement by modifying the neural network architecture. Using the `policy_kwargs` parameter: `dict(net_arch=[256, 256, 128])`, we were able to obtain more satisfactory results for the Cheetah model. This modification allowed the neural network to better learn complex and appropriate behaviors for the task.

- **policy_kwargs**: We configured the neural network architecture with three layers of 256, 256, and 128 neurons respectively, improving the model's learning capacity.

In summary, through the introduction of targeted penalties and modification of neural network architectures, we were able to significantly improve the performance of the Cheetah model.

### 6.2 Challenges with the Ant Model
Regarding the Ant model, we explored various strategies to improve performance. The Ant environment presents greater computational complexity, requiring more accurate optimization of parameters and neural network architecture. We tested various algorithms and adapted control strategies to optimize the model's behavior.

In particular, we refined the neural network architecture and adjusted key parameters to handle the environment's complexity. Despite the inherent difficulties of the task, we were able to identify and implement the best solutions among those tested. This allowed us to make significant progress in training the Ant model, achieving concrete results. However, we recognize that there is still room for improvement to reach a definitive optimal solution.

In summary, through the introduction of targeted penalties and modification of neural network architectures, we were able to significantly improve the performance of the Cheetah and Ant models, although the optimization process continues to be a work in progress.