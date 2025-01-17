# Making it compatible for two GPUs -- mostly how its called
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from utils import get_current_date_time_string
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model_path = os.path.join("models", get_current_date_time_string())
    if rank == 0:
        os.makedirs(model_path, exist_ok=True)

    ENV_NAME = 'SuperMarioBros-1-1-v3'
    SHOULD_TRAIN = True
    DISPLAY = rank == 0  # Only display on main process
    CKPT_SAVE_INTERVAL = 5000
    NUM_OF_EPISODES = 50_000

    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', 
                                  apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n, rank=rank)

    if not SHOULD_TRAIN:
        folder_name = ""
        ckpt_name = ""
        agent.load_model(os.path.join("models", folder_name, ckpt_name))
        agent.epsilon = 0.2
        agent.eps_min = 0.0
        agent.eps_decay = 0.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    for i in range(NUM_OF_EPISODES):    
        if rank == 0:
            print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        
        while not done:
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(a)
            total_reward += reward

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state

        if rank == 0:
            print("Total reward:", total_reward, "Epsilon:", agent.epsilon, 
                  "Size of replay buffer:", len(agent.replay_buffer), 
                  "Learn step counter:", agent.learn_step_counter)

        if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0 and rank == 0:
            model_save_path = os.path.join(model_path, f"model_{i + 1}_iter.pt")
            agent.save_model(model_save_path)
            print(f"Model saved at: {model_save_path}")

        if rank == 0:
            print("Total reward:", total_reward)

    env.close()
    cleanup()

if __name__ == "__main__":
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size)