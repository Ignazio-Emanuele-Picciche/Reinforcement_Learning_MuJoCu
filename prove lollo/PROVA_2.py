# Importa le librerie necessarie
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Crea l'ambiente "Ant-v4"
env = gym.make('Ant-v4')

# Aggiungi il wrapper DummyVecEnv
env = DummyVecEnv([lambda: env])

# Crea il modello PPO con una politica basata su MLP (Multi-Layer Perceptron)
'''
sono stati aggiunti al modello PPO leanring rate nsteps batch size e gamma per poter modificare i parametri del modello

'''
model = PPO("MlpPolicy", env,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,verbose=1)

# Addestra il modello per un certo numero di passi temporali
print("Inizio dell'addestramento...")
model.learn(total_timesteps=10000)  # Puoi cambiare il numero di passi temporali in base alle tue esigenze
print("Addestramento completato.")

# Testa il modello e mostra i dati di miglioramento
print("Inizio del testing...")

episodes = 10  # Numero di episodi da testare
episode_rewards = []  # Lista per registrare le ricompense di ogni episodio

for episode in range(episodes):
    obs = env.reset()  # Resetta l'ambiente all'inizio di ogni episodio
    done = False
    episode_reward = 0  # Ricompensa cumulativa per l'episodio corrente

    while not done:  # Controlla il valore booleano di 'done'
        action, _states = model.predict(obs)  # Predice l'azione basata sull'osservazione
        obs, reward, done, info = env.step(action)  # Applica l'azione all'ambiente
        episode_reward += np.mean(reward)  # Aggiorna la ricompensa cumulativa

    episode_rewards.append(episode_reward)  # Registra la ricompensa totale dell'episodio
    print(f"Episodio {episode + 1}: Ricompensa cumulativa = {episode_reward:.2f}")

# Stampa le statistiche finali
print("\nStatistiche di testing:")
print(f"Ricompensa media: {np.mean(episode_rewards):.2f}")
print(f"Ricompensa massima: {np.max(episode_rewards):.2f}")
print(f"Ricompensa minima: {np.min(episode_rewards):.2f}")

# Chiudi l'ambiente
env.close()