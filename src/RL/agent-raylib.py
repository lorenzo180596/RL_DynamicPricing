import os
import tomli
import ray
from ray import tune, air
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.algorithm import Algorithm
from environment import SimulationEnvironment



# get configs from config file
with open("config/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)

# paths definition
CHECKPOINT_PATH = os.getcwd() + '/checkpoints'

class DDQNAgent():
    def __init__(self):

        config = DQNConfig()
        config.environment(env=SimulationEnvironment)
        config.training(
            gamma=CFG['ddqn']['gamma'],
            lr=CFG['ddqn']['learning_rate'],
            train_batch_size=CFG['ddqn']['batch_size'],
            model={
                'fcnet_hiddens': CFG['ddqn']['fcnet_hiddens'],
                'fcnet_activation': CFG['ddqn']['fcnet_activation']
            },
            num_atoms=CFG['ddqn']['num_atoms'],
            v_min=CFG['ddqn']['v_min'],
            v_max=CFG['ddqn']['v_max'],
            noisy=CFG['ddqn']['noisy'],
            dueling=CFG['ddqn']['dueling'],
            hiddens=CFG['ddqn']['hidden_size'],
            double_q=CFG['ddqn']['double_q'],
            n_step=CFG['ddqn']['update_step']
        )

        config.exploration(
            exploration_config={
                'initial_epsilon': CFG['ddqn']['max_epsilon'],
                'final_epsilon': CFG['ddqn']['min_epsilon']
            }
        )

        config.tau = CFG['ddqn']['tau']
        config.replay_buffer_config['capacity'] = CFG['ddqn']['memory_size']

        self.tuner = tune.Tuner(
            "DQN",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop={"training_iteration": CFG['agent']['n_iterations']},
                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
                local_dir=CHECKPOINT_PATH
            )
        )

    def train(self):

        # train tuner object
        ray.shutdown()
        ray.init()
        results = self.tuner.fit()
        ray.shutdown()

        # select best result
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        best_checkpoint = best_result.checkpoint
        print('Best result: ', best_result)
        print('Best checkpoint: ', best_checkpoint)

    def test(self):
        pass


class RainbowAgent():
    def __init__(self):

        config = DQNConfig()
        config.environment(env=SimulationEnvironment)
        config.training(
            gamma=CFG['rainbow']['gamma'],
            lr=CFG['rainbow']['learning_rate'],
            train_batch_size=CFG['rainbow']['batch_size'],
            model={
                'fcnet_hiddens': CFG['rainbow']['fcnet_hiddens'],
                'fcnet_activation': CFG['rainbow']['fcnet_activation']
            },
            num_atoms=CFG['rainbow']['num_atoms'],
            v_min=CFG['rainbow']['v_min'],
            v_max=CFG['rainbow']['v_max'],
            noisy=CFG['rainbow']['noisy'],
            dueling=CFG['rainbow']['dueling'],
            hiddens=CFG['rainbow']['hidden_size'],
            double_q=CFG['rainbow']['double_q'],
            n_step=CFG['rainbow']['update_step']
        )

        config.exploration(
            exploration_config={
                'initial_epsilon': CFG['rainbow']['max_epsilon'],
                'final_epsilon': CFG['rainbow']['min_epsilon']
            }
        )

        config.tau = CFG['rainbow']['tau']
        config.replay_buffer_config['capacity'] = CFG['rainbow']['memory_size']

        self.tuner = tune.Tuner(
            "DQN",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop={"training_iteration": CFG['agent']['n_iterations']},
                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
                local_dir=CHECKPOINT_PATH
            )
        )

    def train(self):

        # train tuner object
        ray.shutdown()
        ray.init()
        results = self.tuner.fit()
        ray.shutdown()

        # select best result
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        best_checkpoint = best_result.checkpoint
        print('Best result: ', best_result)
        print('Best checkpoint: ', best_checkpoint)

    def test(self):
        pass

class TD3Agent():
    def __init__(self):

        config = TD3Config()
        config.training(
            gamma=CFG['td3']['gamma'],
            lr=CFG['td3']['learning_rate'],
            train_batch_size=CFG['td3']['batch_size'],
            twin_q=CFG['td3']['twin_q'],
            actor_hiddens=CFG['td3']['actor_hiddens'],
            actor_hidden_activation=CFG['td3']['actor_hidden_activation'],
            critic_hiddens=CFG['td3']['critic_hiddens'],
            critic_hidden_activation=CFG['td3']['critic_hidden_activation'],
            tau=CFG['td3']['tau']
        )

        self.tuner = tune.Tuner(
            "TD3",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop={"training_iteration": CFG['agent']['n_iterations']},
                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
                local_dir=CHECKPOINT_PATH
            )
        )

    def train(self):
        
        # train tuner object
        ray.shutdown()
        ray.init()
        results = self.tuner.fit()
        ray.shutdown()

        # select best result
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        best_checkpoint = best_result.checkpoint
        print('Best result: ', best_result)
        print('Best checkpoint: ', best_checkpoint)

    def test(self):
        pass