"""
This module provides the main functionality able to run all the simulation.
Select the kind of simulation you want to perform through the config file.

Example
-------------
Select mode = 1 in the config file to run the training of the DDDQN algorithm.
If load_nn is set to true in the config file it will be requested to import a previously trained neural network to resume training.
Then, simulation will start and will last for the number of episodes set up in the config file. A subfolder in the rst folder will be generated with the output. 

"""

import toml
import numpy as np
import os
import torch 
import json
import sys

from lib import *
from tqdm.auto import tqdm
from collections import deque
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from copy import deepcopy

from storage import Storage
from environment import SimulationEnvironment
from RL.model import ReplayMemory



if __name__ == '__main__':

    ####################################
    # GET CONFIGURATIONs FROM TOMLI FILE
    ####################################
    config_path = get_config_file() 

    with open(config_path) as config_file:
        config = toml.load(config_file)

    if config['simulation']['mode'] == 0:
        training_flag = True
        validation_flag = False
        test_flag = False
    elif config['simulation']['mode'] == 1:
        training_flag = False
        validation_flag = True
        test_flag = False
    elif config['simulation']['mode'] == 2:
        training_flag = False
        validation_flag = False
        test_flag = True
    if training_flag:
        dynamic_our_property = True
    else:
        dynamic_our_property = config['simulation']['dynamic_our_property']  
    if validation_flag:
        load_nn = True
    else:
        load_nn = config['simulation']['load_nn']
    if test_flag:
        draw_charts = True
        save_to_excel = True
    else:
        draw_charts = False
        save_to_excel = False

    ###################################
    # SET NUMPY SEED
    ###################################
    if config['simulation']['use_custom_seed']:
        seed = config['simulation']['custom_seed']
        np.random.seed(seed)
    
    ###################################
    # CREATE PATHS FOR SAVING AND LOAD INFO
    ###################################
    if load_nn:
        model_to_load_path = askopenfilename() #TODO: fix the window (two windows open, one remains opened)
        data_last_training_path = get_data_last_training_file(model_to_load_path)
        with open(data_last_training_path, "rb") as config_file:
            data_last_training = toml.load(data_last_training_path)
    else:
        model_to_load_path = None

    storage = Storage(config= config,
                      training_flag= training_flag,
                      validation_flag= validation_flag,
                      test_flag= test_flag,
                      dynamic_our_property= dynamic_our_property,
                      model_to_load_path= model_to_load_path)

    with open(os.path.join(storage.folder_path, 'config_used.txt'), 'w') as f:
        for first_key in config:
            f.write(f"[{first_key}]: \n")
            for second_key in config[first_key]:
                f.write(f"\t - {second_key}: {config[first_key][second_key]} \n")
    
    ###################################
    # INIT ENVIRONMENT, AGENT AND NN
    ###################################
    which_state = config['models']['state']
    which_agent = config['models']['algorithm']
    which_reward = config['models']['reward']
    dynamic_price_competitors = config['simulation']['dynamic_price_competitors']
    tot_episodes = config['simulation']['tot_episodes']
    which_action = config['models']['action']

    if training_flag:        
        # Define epsilon for every episode
        epsilon_array = np.zeros((tot_episodes))
        if load_nn:
            total_episode_this_NN = data_last_training['total_episode_this_NN']
            max_epsilon = config['ddqn']['min_epsilon'] + (config['ddqn']['max_epsilon'] - config['ddqn']['min_epsilon']) * np.exp(-config['ddqn']['decay_rate']*(total_episode_this_NN))
        else:
            max_epsilon = config['ddqn']['max_epsilon']
            total_episode_this_NN = 0
        for i in range(tot_episodes):
            epsilon = config['ddqn']['min_epsilon'] + (max_epsilon - config['ddqn']['min_epsilon']) * np.exp(-config['ddqn']['decay_rate']*i)
            epsilon_array[i] = epsilon
    elif validation_flag: 
        epsilon_array = [config['ddqn']['min_epsilon'] for _ in range(tot_episodes)]
    elif test_flag:
        epsilon_array = [config['ddqn']['epsilon_test'] for _ in range(tot_episodes)]

    env = SimulationEnvironment(config= config, 
                                dynamic_price_competitors= dynamic_price_competitors, 
                                which_state= which_state, 
                                which_action= which_action, 
                                which_agent = which_agent, 
                                which_reward= which_reward, 
                                dynamic_our_property= dynamic_our_property,
                                draw_charts= draw_charts)
    
    if validation_flag:
        env_validation_no_dynamic = deepcopy(env)
        env.our_property.dynamic_price = True
        env_validation_no_dynamic.our_property.dynamic_price = False
        rev_no_dyn = [0 for _ in range(tot_episodes)]
        rev_dyn = [0 for _ in range(tot_episodes)]
        winner_env = [0 for _ in range(tot_episodes)]
        rev_diff_percentage = [0 for _ in range(tot_episodes)]


    if env.env_settings.discrete:
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n 

    ###################################
    # LOAD NN AND MEMORY BUFFER
    ###################################
    if load_nn:
        if training_flag:
            memory_buffer_path = get_memory_buffer(model_to_load_path)
            with open(memory_buffer_path, "r") as fp:
                array_loaded = json.load(fp)
            for i in range(len(array_loaded)):
                cur_state_to_load = np.array(array_loaded[i][0],dtype= np.float32)
                next_state_to_load = np.array(array_loaded[i][3],dtype= np.float32)
                env.agent.memory.push(cur_state_to_load, 
                                      array_loaded[i][1], 
                                      array_loaded[i][2], 
                                      next_state_to_load, 
                                      array_loaded[i][4])
                
        env.agent.policy_model.load_state_dict(torch.load(model_to_load_path))
        env.agent.policy_model.eval()

    ###################################
    # INIT REWARD
    ###################################   

    total_rewards = []
    rewards_deque = deque(maxlen= config['RL']['rewards_window_size'])
    if load_nn:
        best_avg_reward = float(data_last_training['best_avg_reward'])
    else:
        best_avg_reward = -10000

    ###################################
    # START TRAINING
    ###################################
    with tqdm(total=tot_episodes, file=sys.stdout) as pbar:
        for episode in range(tot_episodes):
            if (validation_flag):
                seed = np.random.randint(0,2**31)
                print(seed)
                np.random.seed(seed)
                sim_episode(epsilon_array, episode, env= env_validation_no_dynamic, agent_active = False)
                np.random.seed(seed)

            state, rewards, epsilon = sim_episode(epsilon_array, episode, env= env)

            if (validation_flag):
                validate_NN(env, env_validation_no_dynamic, episode, rev_dyn, rev_no_dyn, winner_env, rev_diff_percentage)
                 
            if draw_charts:
                compute_cumulative_data_episodes(env)
            if save_to_excel:
                if config['simulation']['use_custom_seed']:
                    save_data_to_excel(episode, env, config, state, n_states, storage, seed)
                else:
                    save_data_to_excel(episode, env, config, state, n_states, storage)
                    
            # update information
            total_rewards.append(rewards)
            rewards_deque.append(rewards)

            # average reward value given the reward window
            avg_rewards = np.mean(rewards_deque)

            # evaluation
            if avg_rewards >= best_avg_reward:
                best_avg_reward = avg_rewards
            if (episode%config['simulation']['how_many_episode_every_save'] == 0 or episode == tot_episodes-1) and not(episode == 0) and training_flag:
                torch.save(env.agent.policy_model.state_dict(), storage.model_path)
                data_for_toml = {
                    "epsilon": epsilon,
                    "best_avg_reward": best_avg_reward,
                    "last_avg_reward": avg_rewards,
                    "last_episode": episode,
                    "total_episode_this_NN":episode+total_episode_this_NN+1
                }
                with open(os.path.join(storage.folder_path, 'data_last_training.toml'), "w") as toml_file:
                    toml.dump(data_for_toml, toml_file)

                memory_to_save = ReplayMemory(capacity=env.agent.memory.size())
                for i in range(env.agent.memory.size()):
                    memory_to_save.push(env.agent.memory.memory[i][0].tolist(), 
                                        int(env.agent.memory.memory[i][1]), 
                                        env.agent.memory.memory[i][2], 
                                        env.agent.memory.memory[i][3].tolist(), 
                                        env.agent.memory.memory[i][4])
                with open(os.path.join(storage.folder_path, 'memory_buffer'), "w") as fp:   #Pickling
                    json.dump(memory_to_save.memory, fp)
                
            # the game is solved by earning more than "max_avg_rewards" for a single episode
            if best_avg_reward > config['RL']['max_avg_rewards']:
                break 

            pbar.set_description(f"Ep. {episode} epsilon: {epsilon:.2f} reward: {rewards:.2f} avg reward: {avg_rewards:.2f} best avg reward: {best_avg_reward:.2f} ")
            pbar.update(1)

    if draw_charts:
        compute_mean_data_episodes(env, episode)
        plot_data(env, storage)

    #Create plot of rewards       
    plt.subplots(figsize=(5, 5), dpi=100)
    plt.plot(total_rewards)
    plt.ylabel('Total Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Total Rewards Per Episode for {} Training Episodes'.format(episode + 1), fontsize=12)
    plt.savefig(storage.plot_training_path, dpi=100, bbox_inches='tight')
    if(validation_flag):
        plot_validation_chart(winner_env, rev_diff_percentage, storage)
    #plt.show()

