import argparse
import time
import copy

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
import sys
import math
from collections import deque
from collections import defaultdict


import utils.utils as utils
from utils.replay_buffer import ReplayBuffer
from utils.constants import *
from models.definitions.DQN import DQN

########################################### Steps to perform for loading the required environment and the dqn model architecture ##########################################

# env = utils.get_env_wrapper(config['env_id'])
# replay_buffer = ReplayBuffer(config['replay_buffer_size'], crash_if_no_mem=config['dont_crash_if_no_mem'])

# utils.set_random_seeds(env, config['seed'])

# linear_schedule = utils.LinearSchedule(
#     config['epsilon_start_value'],
#     config['epsilon_end_value'],
#     config['epsilon_duration']
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule).to(device)

def train_dqn(config):
    env = utils.get_env_wrapper(config['env_id'])
    env.metadata['render_fps'] = 4
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], crash_if_no_mem=config['dont_crash_if_no_mem'])

    utils.set_random_seeds(env, config['seed'])

    linear_schedule = utils.LinearSchedule(
        config['epsilon_start_value'],
        config['epsilon_end_value'],
        config['epsilon_duration']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule).to(device)
    return dqn, env


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--seed", type=int, help="Very important for reproducibility - set the random seed", default=23)
    parser.add_argument("--env_id", type=str, help="Atari game id", default='BreakoutNoFrameskip-v4')
    parser.add_argument("--num_of_training_steps", type=int, help="Number of training env steps", default=10_000_000)
    parser.add_argument("--acting_learning_step_ratio", type=int, help="Number of experience collection steps for every learning step", default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clipping_value", type=float, default=5)  # 5 is fairly arbitrarily chosen

    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=1000000)
    parser.add_argument("--dont_crash_if_no_mem", action='store_false', help="Optimization - crash if not enough RAM before the training even starts (default=True)")
    parser.add_argument("--num_warmup_steps", type=int, help="Number of steps before learning starts", default=50000)
    parser.add_argument("--target_dqn_update_interval", type=int, help="Target DQN update freq per learning update", default=10000)

    parser.add_argument("--batch_size", type=int, help="Number of states in a batch (from replay buffer)", default=32)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--tau", type=float, help='Set to 1 for a hard target DQN update, < 1 for a soft one', default=1.)

    # epsilon-greedy annealing params
    # parser.add_argument("--epsilon_start_value", type=float, default=1.0)
    # parser.add_argument("--epsilon_end_value", type=float, default=0.1)
    # parser.add_argument("--epsilon_duration", type=int, default=1000000)
    parser.add_argument("--epsilon_start_value", type=float, default=0.01)
    parser.add_argument("--epsilon_end_value", type=float, default=0.01)
    parser.add_argument("--epsilon_duration", type=int, default=1)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="Log to console after this many env steps (None = no logging)", default=10000)
    parser.add_argument("--log_freq", type=int, help="Log metrics to tensorboard after this many env steps (None = no logging)", default=10000)
    parser.add_argument("--episode_log_freq", type=int, help="Log metrics to tensorboard after this many episodes (None = no logging)", default=5)
    parser.add_argument("--checkpoint_freq", type=int, help="Save checkpoint model after this many env steps (None = no checkpointing)", default=100000)
    parser.add_argument("--grads_log_freq", type=int, help="Log grad norms after this many weight update steps (None = no logging)", default=2500)
    parser.add_argument("--debug", action='store_true', help='Train in debugging mode')
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


# if __name__ == '__main__':
#     # Train the DQN model
#     model = train_dqn(get_training_args())

########################################## Now we load the DQN model #############################################################################
if __name__ == '__main__':
    # Train the DQN model
    model, env = train_dqn(get_training_args())
    target_model, _ = train_dqn(get_training_args())

# print(model)

# Now we use the instance of dqn named model for all purposes
folder_path = f'/home/grads/r/romils/Downloads/w2v_old/models/checkpoints_new/' # this stored upto 10M time-steps for norm of diff calcs
# folder_path = f'/home/grads/r/romils/Downloads/w2v_old/models/checkpoints/' # this includes all checkpoint stored upto 50M time-steps
checkpoint_path_list = os.listdir(folder_path)

# print(checkpoint_path_list)
'''
Now since the checkpoint_path list is arranged ad-hoc and directly using sorted doesn't work since it treats 1000 and 10000 to be less than 1100,
hence we will manually update the count and search for the corresponding multiple of 100k file
'''
# Sorting the list based on the numeric value at the end of each element
sorted_list = sorted(checkpoint_path_list, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
# print(sorted_list[-1])

norms = [] # this stores the element wise squared difference norm of consecutive models

# for i in range(len(sorted_list)-10):
i = 40
index_arr = np.array([1, 2, 5, 10, 20, 40])
checkpoint_1 = torch.load(folder_path + sorted_list[40])
for ind in index_arr:
    # checkpoint_1 = torch.load(folder_path + sorted_list[i])
    checkpoint_2 = torch.load(folder_path + sorted_list[i+ind])

    model.load_state_dict(checkpoint_1['state_dict'])
    model.to('cuda')
    target_model.load_state_dict(checkpoint_2['state_dict'])
    target_model.to('cuda')

    model.eval()
    target_model.eval()

    d1 = torch.norm(model.state_dict()['cnn_part.0.weight'] - target_model.state_dict()['cnn_part.0.weight'])
    d2 = torch.norm(model.state_dict()['cnn_part.0.bias'] - target_model.state_dict()['cnn_part.0.bias'])

    d3 = torch.norm(model.state_dict()['cnn_part.2.weight'] - target_model.state_dict()['cnn_part.2.weight'])
    d4 = torch.norm(model.state_dict()['cnn_part.2.bias'] - target_model.state_dict()['cnn_part.2.bias'])

    d5 = torch.norm(model.state_dict()['cnn_part.4.weight'] - target_model.state_dict()['cnn_part.4.weight'])
    d6 = torch.norm(model.state_dict()['cnn_part.4.bias'] - target_model.state_dict()['cnn_part.4.bias'])

    d_final = d1+d2+d3+d4+d5+d6

    norms.append(d_final.cpu().numpy())

    # Now we search for the CNN layers and compute the norm of the difference between those weights
np.save('norm_weights_cnn_40_starter.npy', np.array(norms))

plt.plot(range(len(norms)), norms, '-dy')
plt.title("Norm of the Difference between the CNN layer weights starting from 4M time-step")
plt.ylabel('norm-difference')
plt.xlabel('time-steps')
plt.savefig('norm_diff_40_starter.jpg')
# plt.show()



# ############ Loading latest model to check performance ###############
# # checkpoint_1 = torch.load(folder_path + sorted_list[-1])
# # checkpoint_1 = torch.load('/home/grads/r/romils/Downloads/w2v_old/models/binaries/dqn_BreakoutNoFrameskip-v4_000003.pth')
# checkpoint_1 = torch.load('/home/grads/r/romils/Downloads/w2v_old/models/checkpoints/dqn_BreakoutNoFrameskip-v4_ckpt_steps_49800000.pth')
# print("checkpoint max reward = ", checkpoint_1['best_episode_reward'])
# model.load_state_dict(checkpoint_1['state_dict'])
# model.to('cuda')
# model.eval()


# #### Now we render the loaded model:
# t = 2
# episode_reward_list = []
# episode_state_list = []
# episode_length = np.empty((t))

# # env.metadata['render_fps'] = 4

# for i in range(t):
#     episode_length[i] = 0
#     seed = np.random.randint(0, 1000000)
#     print("Current seed = ", seed)
#     current_frame, _  = env.reset(seed = seed)
#     done = False
#     episode_reward = 0
#     old_frames = np.empty((1, 4, 84, 84))
#     new_frames = np.empty((1, 4, 84, 84))
#     # print(frames.shape)
#     old_frames[0,:,:,:] = current_frame
#     # for j in range(3):
#     #     new_frame, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     #     old_frames[0,j+1,:,:] = new_frame
#     #     episode_reward+=reward
#     current_state = torch.from_numpy(old_frames).float()
#     # episode_state_list = [current_state]
#     episode_state_list.append(current_state)
#     current_state = current_state.cuda()
#     # episode_state_list = [current_state]
#     while not done:
#         action = model.epsilon_greedy(current_state)
#         new_frame, reward, terminated, truncated, info = env.step(action)
#         new_frames[0,0,:,:] = new_frame
#         episode_reward+= reward
#         for j in range(3): # repeating the same action 3 more times to get required number of frames
#             new_frame, reward, terminated, truncated, info = env.step(action)
#             new_frames[0,j+1,:,:] = new_frame
#             episode_reward+= reward
#         # end action for
#         done = terminated or truncated
#         current_state = torch.from_numpy(new_frames).float()
#         episode_state_list.append(current_state) # to avoid cuda while saving frames to list
#         current_state = current_state.cuda()

#         # episode_state_list.append(current_state)
#         # env.render()

#         episode_length[i]+=1
    
#     # end while loop
#     episode_reward_list.append(episode_reward)

#     # for i in range(len(episode_state_list)):
#     #     current_state = episode_state_list[i]
#     #     env.render()
#     print("Current Episode length = ", episode_length[i])
#     print("Current Episode reward = ", episode_reward_list[i])


# ##### gif making function
# def save_frames_as_gif(frames, path='./', filename='breakout_sample.gif'): # for creating the frame
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

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=200)
#     # anim.save(path + filename, writer='imagemagick', fps=60)
#     anim.save(path + filename, writer='pillow', fps=60)

# save_frames_as_gif(episode_state_list)


# ####### Checking Temp work: ##############
# # current_frame, _  = env.reset()
# # print(current_frame.shape)
# # print(current_frame)
# # frames = np.empty((1, 4, 84, 84))
# # print(frames.shape)
# # frames[0,0,:,:] = current_frame
# # for i in range(3):
# #     new_frame, reward, terminated, truncated, info = env.step(env.action_space.sample())
# #     frames[0,i+1,:,:] = new_frame

# # print(frames)
# print("Level 4 done")

# # checkpoint = torch.load()
print("Level done!")