# Importa le librerie necessarie
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Crea l'ambiente "Ant-v4"
env = gym.make('Ant-v4', render_mode="human")  # Aggiungi render_mode="human" se vuoi visualizzare l'ambiente

# Aggiungi il wrapper DummyVecEnv
env = DummyVecEnv([lambda: env])

# Crea il modello PPO con una politica basata su MLP (Multi-Layer Perceptron)
model = PPO("MlpPolicy", env, verbose=1)

# Addestra il modello per un certo numero di passi temporali
model.learn(total_timesteps=1000)  # Puoi cambiare il numero di passi temporali in base alle tue esigenze
''' 
aumentare numero di steps sopra
'''

# Testa il modello (senza registrare il video)
obs = env.reset()  # Resetta l'ambiente
for _ in range(1000):
    action, _states = model.predict(obs)  # Predice l'azione basata sull'osservazione
    obs, reward, done, info = env.step(action)  # Applica l'azione all'ambiente

    if done.any():  # Controlla se l'episodio è finito
        obs = env.reset()  # Resetta l'ambiente se l'episodio è finito

# Chiudi l'ambiente
env.close()
