#WITHOUT DIFFERENTIABLE SAFETY LAYER 
#Compatible with the task scheduler (created on 9/5/2025)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import Continuous2DEnv
from datetime import datetime

import sys
import random
import torch
import os
import copy

from arguments import get_args
from SAC import SAC
from evaluate_policy import eval_policy
import tkinter as tk
from tkinter import filedialog

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path


# def train(env,hyperparameters, actor_model, critic_model):
# 	"""
# 		Trains the model.

# 		Parameters:
# 			env - the environment to train on
# 			hyperparameters - a dict of hyperparameters to use, defined in main
# 			actor_model - the actor model to load in if we want to continue training
# 			critic_model - the critic model to load in if we want to continue training

# 		Return:
# 			None
# 	"""	
# 	print(f"Training", flush=True)


# 	model = SAC(env=env, **hyperparameters)

# 	# Tries to load in an existing actor/critic model to continue training on
# 	if actor_model != '' and critic_model != '':
# 		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
# 		model.actor.load_state_dict(torch.load(actor_model))
# 		model.critic.load_state_dict(torch.load(critic_model))
# 		print(f"Successfully loaded.", flush=True)
# 	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
# 		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
# 		sys.exit(0)
# 	else:
# 		print(f"Training from scratch.", flush=True)

# 	model.train()

def train(env,hyperparameters, num_runs, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			num_runs - number of independent runs to average over
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	now = datetime.now()
	formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
	if hyperparameters['CBF']:
		folder_name = 'CBF_' + formatted_time
	else:
		folder_name = 'noCBF_' + formatted_time


	print(f"Training started...", flush=True)

	eps_rewards_all_runs = np.zeros((num_runs, hyperparameters['num_episodes']))
	total_violations = 0

	for i in range(num_runs):
		print(f"\nRun {i+1} / {num_runs}:", flush=True)
		print('-----------------------', flush=True)
		folder_name_run = folder_name + f'/{i+1}'

		hyperparameters['folder_name'] = folder_name_run #folder name to save the model and plots in

		model = SAC(env=env, **hyperparameters)

		# Tries to load in an existing actor/critic model to continue training on
		if actor_model != '' and critic_model != '':
			print(f"Loading in {actor_model} and {critic_model}...", flush=True)
			model.actor.load_state_dict(torch.load(actor_model))
			model.critic.load_state_dict(torch.load(critic_model))
			print(f"Successfully loaded.", flush=True)
		elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
			print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
			sys.exit(0)
		else:
			print(f"Training from scratch.", flush=True)

		eps_rewards_all_runs[i], num_violations = model.train()
		total_violations += num_violations

	save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', folder_name)
	os.makedirs(save_dir, exist_ok=True)

	print('Total constraint violations across all runs:', total_violations)

	np.save(os.path.join(save_dir, "eps_rewards_all_runs.npy"), eps_rewards_all_runs)
	print(f"Saved eps_rewards_all_runs to {save_dir}/eps_rewards_all_runs.npy")

	plot_final_results(eps_rewards_all_runs, folder_name)

def test(env, hyperparameters, model_path, max_timesteps_per_episode):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {model_path}", flush=True)

	#if the actor model is not specified, then exit
	if model_path == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	#extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	model = SAC(env = env, **hyperparameters)

	# Load in the actor model saved by the PPO algorithm
	model.load_model(model_path)

	policy = model.policy_net

	eval_policy(policy=policy, env=env, max_timesteps_per_episode=max_timesteps_per_episode)


def plot_final_results(rewards, folder_name):
	mean = rewards.mean(axis=0)
	std = rewards.std(axis=0, ddof=1)  # sample std
	lo = mean - 2 * std
	hi = mean + 2 * std

	x = np.arange(0, rewards.shape[1])
	plt.close()
	plt.plot(x, mean, linewidth=2)
	plt.fill_between(x, lo, hi, alpha=0.2)
	plt.xlabel('Episode')
	plt.ylabel('Episode Reward')
	plt.grid(True)


	base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
	plot_dir = os.path.join(base_dir, folder_name)
	os.makedirs(plot_dir, exist_ok=True)


	plot_path = os.path.join(plot_dir, 'final.png')
	plt.savefig(plot_path) #save the plot with the current date and time
	print('Plot saved!')
	plt.show()


if __name__ == "__main__":
	agent_init_loc = [-5.0, -5.0] #initial location of the agent
	#Static goal region:
	goal_region_radius = 2
	goals = {
	0: {'center': (6, 5), 'radius': goal_region_radius, 'movement':{'type':'static'}}, #goal region for the agent
	}

	obstacles = {
	0: {'label': 1,'center': (3, -3), 'radius': 1.5, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'},
	1: {'label': 1,'center': (-3, 3), 'radius': 1.5, 'u_max': 1, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'}
	}


	u_agent_max = 0.5 # Max vel. of the system
	disturbance_interval = [0, 0]


    #config dictionary for the environment
	config = {
        'init_loc':agent_init_loc, #initial location of the agent (x, y)
        "width": 10.0,
        "height": 10.0,
        "dt": 1,
        "render": False,
		'dt_render': 0.03,
		'goals': goals, #goal regions for the agent
        "obstacles": obstacles, #obstacle regions in the environment
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "single integrator", #dynamics model to use
		"targets": None, #target regions for the agent to visit
		"disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }

	action_range = [0.5, 0.5] #action range for the RL model (for the neural network output layer) [3,3] for case-1, [4,4] for case-2

	#learning hyperparameters:
	hyperparameters = {
				'gamma': 0.99,
				'tau': 0.005,
				'hidden_size': 64, 
				'buffer_size': int(1e6),
				'batch_size': 128,
				'max_timesteps_per_episode': 35, 
				'num_episodes': 100,
				'n_updates_per_iteration': 1,
				'deterministic': False,
				'auto_entropy':True,
				'action_range': action_range,
				'action_clip' : True,
				'CBF': False,
				'CBF_params': None,
				'online': False,
				'disturbance': disturbance_interval
			  }


	args = get_args()

	env = Continuous2DEnv(config)

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		num_runs = 1 #number of independent runs to average over
		train(env=env, hyperparameters=hyperparameters, num_runs=num_runs, actor_model='', critic_model='')

	elif args.mode == 'test':
		config['render'] = True #enable rendering for testing
		config['dt_render'] = 0.03
		config['init_loc'] = [-5.0, -5.0]
		config['randomize_loc'] = False #randomize the agent location at the end of each episode
		env = Continuous2DEnv(config)
		max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
		# Load in the model file
		model_path= select_model_file()
		test(env=env, hyperparameters=hyperparameters, model_path=model_path, max_timesteps_per_episode=max_timesteps_per_episode)
