import gymnasium as gym
import numpy as np
import random
import imageio.v2 as imageio
from evolution.neural_network import NeuralNetwork


def evaluate_genome(genome, input_size, hidden1_size, hidden2_size, output_size, n_genome, generation, seeds, render=False):
    total_rewards = []

    env = gym.make("LunarLander-v3", render_mode="rgb_array" if (render and (generation == 0 or generation % 100 == 0)) else None)
    nn = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size, genome)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        frames = []

        while not done:
            if render and (generation == 0 or generation % 100 == 0):
                frame = env.render()
                frames.append(frame)

            output = nn.forward(obs)
            action = np.argmax(output)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if total_reward < -200:
                total_reward = -300
                break

        total_rewards.append(total_reward)

        if render and (generation == 0 or generation % 100 == 0):
            imageio.mimsave(f"assets/gifs/lunarlander_{generation}.{n_genome}.{seed}_{total_reward:.2f}.gif", frames, fps = 45, loop=0)

    env.close()

    return sum(total_rewards) / len(total_rewards)
