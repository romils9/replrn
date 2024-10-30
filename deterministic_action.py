# import argparse
# import time
# import copy

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg') # Non-GUI backend

import matplotlib.pyplot as plt
from matplotlib import animation
# from torch.optim import Adam
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter   
import os
# import sys
# import math
from collections import deque
from collections import defaultdict


import utils.utils as utils
from utils.replay_buffer import ReplayBuffer
from utils.constants import *
from models.definitions.DQN import DQN

import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper

################################################ Set up the wrapper #############################################################
class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_shape = np.roll(self.observation_space.shape, shift=1)  # shape: (H, W, C) -> (C, H, W)

        # Update because this is the last wrapper in the hierarchy, we'll be pooling the env for observation shape info
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)  # shape: (H, W, C) -> (C, H, W)

def get_env_wrapper(env_id, record_video=False):
    """
        Ultimately it's not very clear why are SB3's wrappers and OpenAI gym's copy/pasted code for the most part.
        It seems that OpenAI gym doesn't have reward clipping which is necessary for Atari.

        I'm using SB3 because it's actively maintained compared to OpenAI's gym and it has reward clipping by default.

    """
    monitor_dump_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'gym_monitor')

    # This is necessary because AtariWrapper skips 4 frames by default, so we can't have additional skipping through
    # the environment itself - hence NoFrameskip requirement
    assert 'NoFrameskip' in env_id, f'Expected NoFrameskip environment got {env_id}'

    # The only additional thing needed, on top of AtariWrapper,
    # is to convert the shape to channel-first because of PyTorch's models
    #env_wrapped = Monitor(ChannelFirst(AtariWrapper(gym.make(env_id))), monitor_dump_dir, force=True, video_callable=lambda episode: record_video)
    env_wrapped = ChannelFirst(AtariWrapper(gym.make(env_id, render_mode='rgb_array')))

    return env_wrapped

## Set the env:
env = get_env_wrapper("BreakoutNoFrameskip-v4")
env.metadata['render_fps'] = 4

#### Now we render the loaded model: 
# env = gym.make("ALE/Breakout-v5", render_mode="human")
# env = gym.make("BreakoutNoFrameskip-v4", render_mode='rgb_array')
t = 1
episode_reward_list = []
episode_state_list = []
episode_frame_list = []
episode_length = np.empty((t))
# A = np.array([3, 0, 0, 2,  0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 0, 2])
A = np.array([3, 0, 2,  0, 2, 0, 3, 0, 3, 0, 2, 0, 2])
len_A = len(A)
print("len_A: ", len_A)

# env.metadata['render_fps'] = 4

for i in range(t):
    episode_length[i] = 0
    seed = np.random.randint(0, 1000000)
    # seed = 319177   
    print("Current seed = ", seed)
    current_frame, _  = env.reset(seed = seed)
    done = False
    episode_reward = 0
    old_frames = np.empty((1, 4, 84, 84))
    new_frames = np.empty((1, 4, 84, 84))
    # print(frames.shape)
    old_frames[0,0,:,:] = current_frame
    count = 0
    action = A[count]
    episode_frame_list.append(current_frame)

    for j in range(3):
        action
        new_frame, reward, terminated, truncated, info = env.step(action)
        old_frames[0,j+1,:,:] = new_frame
        episode_reward+=reward
        episode_frame_list.append(new_frame)
    current_state = torch.from_numpy(old_frames).float()

    episode_state_list.append(current_state)
    # current_state = current_state.cuda()
    count+=1

    while not done:
        action = A[count]
        new_frame, reward, terminated, truncated, info = env.step(action)
        new_frames[0,0,:,:] = new_frame
        episode_reward+= reward
        episode_frame_list.append(new_frame)
        for j in range(3): # repeating the same action 3 more times to get required number of frames
            new_frame, reward, terminated, truncated, info = env.step(action)
            new_frames[0,j+1,:,:] = new_frame
            episode_reward+= reward
            episode_frame_list.append(new_frame)
        # end action for
        count +=1
        done = terminated or truncated
        current_state = torch.from_numpy(new_frames).float()
        episode_state_list.append(current_state) # to avoid cuda while saving frames to list
        # current_state = current_state.cuda()

        # episode_state_list.append(current_state)
        env.render()

        episode_length[i]+=1
        count = count % len_A # this ensures the cycle continues if the actions don't end the game
    
    # end while loop
    episode_reward_list.append(episode_reward)

    # for i in range(len(episode_state_list)):
    #     current_state = episode_state_list[i]
    #     env.render()
    print("Current Episode length = ", episode_length[i])
    print("Current Episode reward = ", episode_reward_list[i])


##### gif making function
# def save_frames_as_gif(frames, path='./', filename='breakout_fixed_actions_2.gif'): # for creating the frame
#     # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
#     plt.figure(figsize=(12, 12), dpi=72)
#     print("Total number of frames = ", len(frames))
#     print("Shape of list of frames = ", np.array(frames).shape)

#     # print(frames[0][0][0].shape)
#     # print("\n\n")
#     patch = plt.imshow(frames[0][0][0])
#     plt.axis('off')

#     def animate(i):
#         if i<4:
#             patch.set_data(frames[i][0][0])

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval= 1000)
#     # anim.save(path + filename, writer='imagemagick', fps=60)
#     anim.save(path + filename, writer='pillow', fps = 1)

def save_frames_as_gif(frames, path='./', filename='breakout_sample_actions_v3.gif'):
    # Set up the figure
    plt.figure(figsize=(12, 12), dpi=72)
    print("Total number of frames =", len(frames))
    print("Shape of list of frames =", np.array(frames).shape)

    # Initialize the first frame
    patch = plt.imshow(frames[0][0][0])
    plt.axis('off')

    # Flatten frames to include all 4 subframes in each entry
    flattened_frames = [frames[i][0][j] for i in range(len(frames)) for j in range(4)]

    # Update function to animate each frame in flattened list
    def animate(i):
        patch.set_data(flattened_frames[i])

    # Create the animation over the flattened frames list
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(flattened_frames), interval=250
    )

    # Save the animation with adjusted fps
    anim.save(path + filename, writer='pillow', fps=5)



# def new_gif(frames, path='./', filename='breakout_fixed_actions_3.gif'):
#     plt.figure(figsize=(12, 12), dpi=72)
#     print("Total number of frames =", len(frames))
#     print("Shape of list of frames =", np.array(frames).shape)

#     # Initialize the first frame
#     patch = plt.imshow(frames[0][0][0])
#     plt.axis('off')

#     # Function to update each frame, displaying every 4th frame
#     def animate(i):
#         frame_index = i * 4  # Skip to every 4th frame
#         if frame_index < len(frames):
#             patch.set_data(frames[frame_index][0][0])

#     # Create animation using only 1 frame in every 4
#     anim = animation.FuncAnimation(
#         plt.gcf(), animate, frames=len(frames) // 4, interval=250
#     )

save_frames_as_gif(episode_state_list)
# new_gif(episode_state_list)

# save_frames_as_gif(episode_frame_list)


####### Checking Temp work: ##############

print("Level 4 done")

# for ind in range(100):
#     print(env.action_space.sample())
