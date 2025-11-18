import math
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from IPython.display import clear_output
from IPython.display import display

from networks import PolicyNetwork, SoftQNetwork, ValueNetwork
import os
from datetime import datetime
import matplotlib.pyplot as plt
from Sequential_CBF_new import *
from CLF import solve_clf_qp, clf_function
import copy



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)



class SAC:

    def __init__(self, env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim, self.hidden_size, self.device, self.action_range).to(self.device)
        self.soft_q_net1 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.target_soft_q_net1 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(self.obs_dim, self.act_dim, self.hidden_size).to(self.device)

        # Initialize target networks with the same weights as the original networks
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.policy_optim = Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.soft_q_optim1 = Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optim2 = Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()


        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.logger = {
            'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations so far
            'eps_rewards': [],
            'actor_losses': [],
            'critic_losses': []
            }


    def _init_hyperparameters(self, hyperparameters):

        #SAC hyperparameters:
        self.soft_q_lr = hyperparameters.get("soft_q_lr", 3e-4)
        self.policy_lr = hyperparameters.get("policy_lr", 3e-4)
        self.alpha_lr = hyperparameters.get("alpha_lr", 3e-4)
        self.hidden_size = hyperparameters.get("hidden_size", 256)

        self.gamma = hyperparameters.get("gamma", 0.99)
        self.tau = hyperparameters.get("tau", 0.005)
        #self.alpha = hyperparameters.get("alpha", 0.2)
        self.replay_buffer_size = hyperparameters.get("buffer_size", int(1e6))
        self.batch_size = hyperparameters.get("batch_size", 256)
        self.max_timesteps_per_episode = hyperparameters.get("max_timesteps_per_episode", 1000)
        self.num_episodes = hyperparameters.get("num_episodes", 1000)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 1)
        self.deterministic = hyperparameters.get("deterministic", False) #whether to use deterministic policy or not during training
        self.auto_entropy = hyperparameters.get("auto_entropy", True) #whether to use automatic entropy tuning or not

        #
        self.action_range = hyperparameters.get("action_range", None) #max speed and max turning rate

        self.online = hyperparameters.get("online", False) #whether to use online learning or not
        
        #CBF parameters:
        self.CBF = hyperparameters.get("CBF", False) #whether to use CBF or not
        self.CBF_params = hyperparameters.get("CBF_params", None) #CBF parameters, if any
        
        self.disturbance = hyperparameters.get("disturbance") #disturbance range in both x and y directions [w_min, w_max]
        self.folder_name = hyperparameters.get("folder_name", None) #folder name to save the model and plots in

        if self.disturbance is not None:
            self.w_max = max(abs(self.disturbance[0]), abs(self.disturbance[1]))


    def update(self, batch_size, auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        #reward = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action), self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optim1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optim1.step()                                

        self.soft_q_optim2.zero_grad()                     
        q_value_loss2.backward()
        self.soft_q_optim2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # Logging
        self.logger['actor_losses'].append(policy_loss.item())
        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()


    def train(self):
        explore_steps = 0 #number of steps to explore before using the policy
        frame_idx = 0 #num of frames so far
        infeasible_solutions = 0 #number of infeasible solutions found during training

        initial_targets = [(key, copy.deepcopy(value)) for key, value in self.CBF_params['targets'].items()]
        u_agent_max = self.CBF_params['u_agent_max']

        #NEW:
        if self.disturbance is not None:
            u_agent_max = u_agent_max - self.w_max #reduce the max agent speed by the disturbance bound (worst-case)

        global_t = 0

        targets = self.CBF_params['targets']
        simulation_targets = self.CBF_params['sim_targets']
        task_violation = 0 #number of episodes where the STL task is violated

        initial_remaining_times = {i: targets[i]['remaining_time'] for i in targets}  #dictionary to store the initial remaining times for each target region


        for eps in range(self.num_episodes):
            
            #reset the targets (episodic STL tasks)
            for target in initial_targets:
                targets[target[0]] = simulation_targets[target[0]]
                targets[target[0]]['remaining_time'] = initial_remaining_times[target[0]] #reset the remaining time for each target region

                
            

            if self.online and eps > 0: #if online learning, go to the next initial position using CLFs
                radius = 5
                x_target = np.random.uniform(-self.env.width, self.env.width)
                y_target = np.random.uniform(-self.env.height, self.env.height)
                target_center = np.array([x_target, y_target])
                alpha = 1.5
                print('Moving to initial location...')
                while np.linalg.norm(target_center - state) - radius >= 0.0:
                    action = solve_clf_qp(state, target_center, alpha, u_agent_max/2)
                    next_state, reward, done = self.env.step(action)
                    state = next_state

                print('Learning continues...')

            else:
                state =  self.env.reset()

            episode_reward = 0

            step = 0 #reset step counter for each episode
             
            while step <= self.max_timesteps_per_episode:
                #print(targets)
                if frame_idx < explore_steps:
                    action_RL = self.policy_net.sample_action()
                    print('Exploration action:', action_RL)
                else:
                    action_RL = self.policy_net.get_action(state, deterministic = self.deterministic)

                    if self.CBF and len(targets) != 0: #If CBF is activated and there are still target regions remaining:
                        #First, calculate the CBF value for each target region:
                        cbf_values = {}
                        for target_index in targets.keys():
                            cbf_value = sequential_CBF(state, u_agent_max, targets, target_index)
                            cbf_values[target_index] = cbf_value

                        min_key = min(cbf_values, key=cbf_values.get)  #find the target region with the minimum CBF value

                        #Now solve the QP to get the control input for the target region with the minimum CBF value:
                        action_CBF = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, step, targets, action_RL)

                        if action_CBF is None:
                            # If the QP fails, we can either ignore the CBF action or set it to zero
                            action= (0.0, 0.0)
                            print('CBF action is None, using zero action instead.')
                            infeasible_solutions += 1 #increment the number of infeasible solutions
                        else:
                            action = (action_RL[0] + action_CBF[0] , action_RL[1] + action_CBF[1]) #a_rl + a_cbf

                    else:
                        action = action_RL
                #print('Action:', action)
                next_state, reward, done = self.env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if self.CBF and len(targets) != 0: #If CBF is activated and there are still target regions remaining:
                    #decrease the remaining time and update the center for each target region
                    for target_index in list(targets.keys()):
                        first_key = next(iter(targets))
                        #print("Target index:", target_index, "Remaining time:", targets[target_index]['remaining_time'])
                        targets[target_index]['remaining_time'] -= 1
                        #targets[target_index]['center'] = moving_target(t, center_of_rotation, targets[target_index]['u_max'])
                        task_type = targets[target_index]['type']
                        time_window = targets[target_index]['time window']

                        #calculate the signed distance to each target region:
                        target_center = targets[target_index]['center']
                        target_radius = targets[target_index]['radius']
                        dist = np.linalg.norm(state[:2] - target_center)
                        signed_distance = dist - target_radius

                        remove_target = False #flag to indicate whether to remove the visited target region

                        if task_type == "F":
                            # Handle F type tasks
                            a = time_window[0][0]
                            b = time_window[0][1]
                            if step >= a and step <= b and signed_distance <= 0:
                                #within the time window
                                remove_target = True
                            else:
                                continue #do not remove the target region if the agent is outside the time window

                        elif task_type == "G":
                            # Handle G type tasks
                            a = time_window[0][0]
                            b = time_window[0][1]
                            if step == a and signed_distance <= 0:
                                remove_target = True #remove the target region if the agent is inside it at the start of the time window
                                #Hold inside the target region until the end of the time window:
                                targets[target_index]['remaining_time'] = 0 #set the remaining time to the length of the time window
                                for j in range(b-a):
                                    #print("Holding inside target region", targets[target_index]['id'], "at time", step)
                                    step += 1

                                    for key in targets.keys():
                                        if key != target_index: #do not decrement the remaining time for the current target region
                                            targets[key]['remaining_time'] -= 1

                                    u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, step, targets, action_RL)

                                    action = (u_cbf[0] + action_RL[0], u_cbf[1] + action_RL[1])  # Combine CBF and RL actions

                                    next_state, reward, done = self.env.step(action) 

                                    self.replay_buffer.push(state, action, reward, next_state, done)

                                    state = next_state
                                    episode_reward += reward          
                            else:
                                continue
                        elif task_type == "FG":
                            # Handle FG type tasks
                            a = time_window[0][0]
                            b = time_window[0][1]
                            c = time_window[1][0]
                            d = time_window[1][1]
                            if step >= (a+c) and step <= (b+c) and signed_distance <= 0:
                                #within the time window
                                remove_target = True
                                targets[target_index]['remaining_time'] = 0 #set the remaining time to the length of the time window
                                for j in range(d-c):
                                    #print("Holding inside target region", targets[target_index]['id'], "at time", step)
                                    step += 1

                                    for key in targets.keys():
                                        if key != target_index: #do not decrement the remaining time for the current target region
                                            targets[key]['remaining_time'] -= 1

                                    u_cbf = solve_cbf_qp(sequential_CBF, state, u_agent_max, min_key, step, targets, action_RL)

                                    action = (u_cbf[0] + action_RL[0], u_cbf[1] + action_RL[1])  # Combine CBF and RL actions

                                    next_state, reward, done = self.env.step(action)

                                    self.replay_buffer.push(state, action, reward, next_state, done)

                                    state = next_state
                                    episode_reward += reward    
                            else: continue

                        elif task_type == "GF":
                            # Handle GF type tasks
                            a = time_window[0][0]
                            b = time_window[0][1]
                            c = time_window[1][0]
                            d = time_window[1][1]
                            if step >= a + c and step <= b + d and signed_distance <= 0 and first_key == target_index: #only remove the target region if it is the first in the sequence
                                #within the time window
                                remove_target = True
                                ##################################################################################
                                #UPDATE the remaining time for the next target region with the same id (if any):
                                ##################################################################################
                                keys = list(targets.keys())
                                this_id = targets[target_index]['id']
                                next_key = None
                                i = keys.index(target_index)
                                for kk in keys[i+1:]:              # look ahead
                                    if targets[kk]['id'] == this_id:
                                        next_key = kk              # found the next duplicate
                                        break
                                if next_key is not None:
                                    targets[next_key]['remaining_time'] = d - c #set the next iteration's remaining time to the second time window's length
                                    #print("Updated remaining time for target", targets[next_key]['id'], "to", targets[next_key]['remaining_time'])
                            else: continue
                        
                        #remove the target if the conditions are satisfied:
                        if remove_target:
                            #print("Agent is inside target region:", targets[target_index]['id'])
                            targets.pop(target_index)  # Remove target region if the agent is inside it

                frame_idx += 1
                step += 1 #increment step counter
                global_t += 1 #increment global time step

                # for target_index in targets.keys():
                #     init_rem_t = initial_remaining_times[target_index]
                #     targets[target_index]['remaining_time'] = init_rem_t - step

                if len(self.replay_buffer) > self.batch_size:
                    for i in range(self.n_updates_per_iteration):
                        _=self.update(self.batch_size, auto_entropy=self.auto_entropy, target_entropy = -1. * self.act_dim)

                # if done: #terminate the episode if target region is reached
                #     break

            if len(targets) != 0: #if the sequence is non-empty after the episode, count it as a task violation
                task_violation += 1


            if eps % 20 == 0 and eps>0: # plot and model saving interval
                    self.save_model(self.folder_name)


            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            self.logger['eps_rewards'].append(episode_reward)
            #self._log_summary()

        #save the final model:
        self.save_model(self.folder_name)

        # plot the results and save the graph:
        plt.close()
        plt.plot(self.logger['eps_rewards'])
        plt.xlabel('Iteration Number')
        plt.ylabel('Episodic Return')
       #plt.title(f'CBF: {self.CBF}, Violations: {self.safety_violations}, Iterations: {len(self.logger["avg_batch_rews"])}, Updates per Iterations: {self.n_updates_per_iteration}')
        plt.grid()

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
        plot_dir = os.path.join(base_dir, self.folder_name)
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, 'plot.png')
        plt.savefig(plot_path, dpi = 200) #save the plot with the current date and time
        print('Plot saved!')

        #plt.show()

        return self.logger['eps_rewards'], task_violation


    def save_model(self, folder_name):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', folder_name)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.soft_q_net1.state_dict(), os.path.join(model_dir, 'Q1.pth'))
        torch.save(self.soft_q_net2.state_dict(), os.path.join(model_dir, 'Q2.pth'))
        torch.save(self.policy_net.state_dict(), os.path.join(model_dir, 'policy.pth'))
        print('Model saved!')


    def load_model(self, path):
        # self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        # self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path))

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        self.policy_net.eval()


    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        i_so_far = self.logger['i_so_far']
        #avg_ep_lens = np.mean(self.logger['eps_lens'])
        ep_rew = self.logger['eps_rewards'][-1]
        actor_loss = self.logger['actor_losses'][-1] #take the last actor loss

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        #print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Episode Return: {ep_rew}", flush=True)
        print(f"Actor Loss: {actor_loss}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
