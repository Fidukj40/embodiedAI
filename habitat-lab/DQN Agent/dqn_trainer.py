import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from habitat_baselines.utils.common import Flatten

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.rollout_storage import RolloutStorage

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# Reference Material: Hands-On Reinforcement Learning for Games By Micheal Lanham:
# Section 2, Constructing DQN and the replay buffer

from collections import deque
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        print('----- init DQN:  num_inputs =',num_inputs,' num_actions =',num_actions)
        self.num_actions = num_actions
        '''
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=16,stride=8),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=8,stride=4),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            Flatten()
        )'''        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=8,stride=4),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        x1 = self.cnn(x)
        x2 = self.fc(x1)
        return x2

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            #state = state.view(state.shape[0],-1)
            #print('------ DQN act state shape = ', state.shape)
            q_value = self.forward(state)
            print('q_value = ', q_value)
            action = torch.argmax(q_value)
        else:
            action = random.randrange(self.num_actions)
        return action



class JUSTGO(nn.Module):
    def __init__(self, num_actions):
        super(JUSTGO, self).__init__()
        print('----- Direct Line of Sight Controller:, num_actions =',num_actions)
        self.num_actions = num_actions

    def forward(self, x):
        return random.choice([0,2,3]) 
        #return random.randrange(2,4)#self.num_actions)

    def act(self, state, epsilon):
        if state[1]<0.2:
          return 1
        if state[0]>0:
          action = 0
        else:
          action = self.forward(state)
        return action


@baseline_registry.register_trainer(name="dqn")
class DQNTrainer(BaseRLTrainer):
    r"""Trainer class for DQN algorithm
    """
    supported_tasks = ["Nav-v0"]
    def __init__(self, config=None, gamma=0.99):
        super().__init__(config)
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        
        self.action_list = list(self.envs.action_spaces[0].spaces.keys())
        print('------------self.action_items = ', self.action_list)
        self.act_space = self.envs.action_spaces[0].n
        observation_dim = self.envs.observation_spaces[0].spaces['depth'].shape
        observation_space = np.prod(observation_dim)
        self.obs_space = observation_space
        
        model = DQN(self.obs_space, self.act_space).to(self.device)
        #model = JUSTGO(self.act_space).to(self.device)

        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters())
        self.replay_buffer = ReplayBuffer(10000)


    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        print(' ************** compute_td_loss  state shape = ', state.shape)
        #state = state.view(state.shape[0],-1)
        #next_state = next_state.view(next_state.shape[0],-1)

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(' ******* compute_td_loss return loss = ', loss)
        return loss


    def train(self, batch_size=128, num_frames=200000):
        losses = []
        all_rewards = []
        episode_reward = 0
        num_success = 0
        soft_spl_score = []
        epsilon_start, epsilon_final = 1.0, 0.1
        epsilon_decay = 50000 #500
        epsilon_schedule = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


        init_observation = self.envs.reset()
        #print(' init_observation = ', init_observation)
        state = init_observation[0]['depth'][::8,::8,:]

        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_schedule(frame_idx)
            if frame_idx>1 and dist_goal<0.2:
              action_idx = 1
            elif isinstance(self.model, DQN):
              action_idx = self.model.act(state, epsilon)
            elif isinstance(self.model, JUSTGO):
              if frame_idx==1:
                action_idx = self.model.act([0,100], epsilon)
              else:
                action_idx = self.model.act([reward,dist_goal], epsilon)
            action = self.action_list[action_idx]

            #next_state, reward, done, _ = self.envs.step([action])
            envstep_outputs = self.envs.step([action])
            observations, rewards, dones, infos = [list(x) for x in zip(*envstep_outputs)]
            
            print('loopIdx =', frame_idx, ' epsilon = ', epsilon, ' is_done = ', dones, ' num_success =',num_success, ' action =',action,'  rewards =', rewards)
            print('pointgoal_GPScompass(unobserved) = ', observations[0]['pointgoal_with_gps_compass'], '  Avg SoftSPL =', np.mean(soft_spl_score))
            print(type(self.model).__name__,'agent Info: ', infos)
            success, dist_goal, sspl = infos[0]['success'], infos[0]['distance_to_goal'], infos[0]['softspl']
            if success > 0.0:
              print(' !!!!!!!!!!!!!   success = ', success)
              num_success += 1
            next_state = observations[0]['depth'][::8,::8,:]

            #print('state shape = ', next_state.shape)
            reward, done = rewards[0] , dones[0]
            self.replay_buffer.push(state, action_idx, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done and len(self.replay_buffer) > batch_size:
                reset_observation = self.envs.reset()
                state = reset_observation[0]['depth'][::8,::8,:]
                all_rewards.append(episode_reward)
                episode_reward = 0
                soft_spl_score.append(sspl)
                if isinstance(self.model, DQN):
                    loss = self.compute_td_loss(batch_size)
                    losses.append(loss.data)

        print('-----------------Finished training: Saving Data --------------- ')
        print('all_rewards len = ', len(all_rewards))
        print('soft_spl len = ', len(soft_spl_score), '   Avg soft_spl = ', np.array(soft_spl_score).mean())
        print('losses len = ', len(losses))

        all_rewards = np.array(all_rewards)
        soft_spl_score = np.array(soft_spl_score)
        losses = np.array(losses)
        #np.savetxt('losses.txt', losses)
        #np.savetxt('all_rewards.txt', all_rewards)
        #np.savetxt('soft_spl_score.txt', soft_spl_score)

