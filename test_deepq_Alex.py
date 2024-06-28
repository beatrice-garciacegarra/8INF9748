import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Initialisation de l'environnement Atari
env_id = 'ALE/Othello-v5'

env = gym.make(env_id)

# Ajout des wrappers pour prétraiter les images
env = AtariWrapper(env, frame_skip=4)

# Modèle
model = DQN(
    "CnnPolicy",  # Politique CNN
    env,
    learning_rate=0.0001,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    #tensorboard_log="./dqn_breakout_tensorboard/",
    verbose=1
)

# Entraînement du modèle
model.learn(total_timesteps=1000000, progress_bar=True)
# Évaluation du modèle
# Runs policy for n_eval_episodes episodes and returns average reward.
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Sauvegarde du modèle
model.save("dqn")

# Chargement du modèle
model = DQN.load("dqn", env=env)

# Visualisation de l'agent entraîné
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    if done:
      obs = vec_env.reset()