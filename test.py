# test.py
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import os
from agent import Agent
import numpy as np

def load_and_test_model(agent, model_path, env, num_episodes=5):
    print(f"Loading model from: {model_path}")
    agent.load_model(model_path)
    agent.epsilon = 0.0  # Disable exploration for testing
    for episode in range(num_episodes):
        done = False
        state, _ = env.reset()
        state = np.repeat(state[:, :, np.newaxis], 4, axis=2)  # Adjust this line to match the input expectation
        total_reward = 0
        while not done:
            state = state.copy()
            state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            new_state = np.repeat(new_state[:, :, np.newaxis], 4, axis=2)  # Adjust this line to match the input expectation
            total_reward += reward
            state = new_state
        print(f"Episode {episode + 1} - Total reward: {total_reward}")

def main():
    model_directory = "models/2024-06-09-00_58_08/"
    model_files = sorted([f for f in os.listdir(model_directory) if f.endswith('.pt')], key=lambda x: int(x.split('_')[1]))

    ENV_NAME = 'SuperMarioBros-1-1-v3'
    DISPLAY = True

    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, RIGHT_ONLY)

    agent = Agent(input_dims=(4, 240, 256), num_actions=env.action_space.n)  # Adjusted input dimensions to 4 channels

    for model_file in model_files:
        model_path = os.path.join(model_directory, model_file)
        load_and_test_model(agent, model_path, env)

    env.close()

if __name__ == "__main__":
    main()
