
"""
Created on Fri Dec  8 2023

@authors: Qiao Wang, Swarna Chetty, Ahmed Al-Tahmeesschi 
"""



#-------------------------Log updates-----------------------------------#
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from collections import namedtuple, deque
import random
import math
import Environment_scenario_0
import datetime
import matplotlib.pyplot as plt
import pickle

import csv

def log_edit(description):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp}: {description}\n"

    with open("DQN_edit_log.txt", "a") as log_file:
        log_file.write(log_entry)


log_edit("Initial version created by Qiao on Fri Dec  1 10:13:00 2023")

log_edit("Updated by Swarna")
# print(log_edit)
#-----------------------------------------------------------------------#


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
GAMMA = 0.99

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1500000
# TAU = 0.005
TAU = 0
LR = 1e-4
MAX_STEP = 100
MAX_EPISODE = 30000
C = 50


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

def select_action(state):
    """
    Action selection: Decaying epsilon greedy
    """
    global steps_done
    global eps_threshold
    
    sample = random.random()
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #    math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = eps_threshold - 1/(MAX_STEP* MAX_EPISODE)
    if eps_threshold<0.1:
        eps_threshold = 0.1
    steps_done += 1
    #print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.choice(action_space)]], device=device, dtype=torch.long)


def optimize_model():
    """
    Optimize the DQN model
    """
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


def play(env, avg_loss_episode, loss_per_step, avg_reward_episode, reward_per_step):
    init_network_states = env.get_state()
    current_state = get_net_states(init_network_states)
    current_state = torch.tensor(current_state, dtype=torch.float32,
                                 device=device).unsqueeze(0)
    current_state = norm_tensor(current_state)

    n_step = 1
    sum_loss = 0
    sum_reward = 0
    reward_per_episode = []
    while n_step <= MAX_STEP:
        # print(current_state)
        #print("step: ", n_step)
        action = select_action(current_state)
        # print("action:",action)

        next_state, reward, complete = execute_action(action.item(), env)
        sum_reward += reward
        
        next_state = get_net_states(next_state)
        reward = torch.tensor([reward], device=device)
        #print("reward: ",reward)

        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = norm_tensor(next_state)
        # print(next_state)

        # Store the transition in bbbb
        memory.push(current_state, action, next_state, reward)
        # print(len(memory))
        # move to the next state
        current_state = next_state

        # Perform one step of the optimization (on the policy network)
        loss_step = optimize_model()
        loss_per_step.append(loss_step)
        sum_loss += loss_step
        
        
        # print(Loss)

        # update target network
        # We may need to set a constant C to control the frequency of updating
        # the target network. TBD
        if steps_done % C == 0:
            #print("Updating target network")
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        n_step += 1
    avg_loss_episode.append(sum_loss / MAX_STEP)
    
    avg_reward_episode.append(sum_reward / MAX_STEP)
    
    
    return 

def get_net_states(network_states):
    """
    Convert network state to DQN state
    Input:
        network_states: {"stations": [dictionaries], "users": [dictionaries]}
    Output:
        DQNstate: size --- numOfUEs * (2 + numOfRUs * 2)
    """
    DQNstate = []
    RSS = []
    uePos = []
    for user_info in network_states['users']:
        uePos = [user_info['location'][0], user_info['location'][1]]
        RSS = user_info['received_power']
        #DQNstate = DQNstate + uePos + RSS
        DQNstate = DQNstate + RSS
    return DQNstate


def execute_action(action_ind, env):
    """
    Execute the action
    Return if the execution is successful
    """
    next_state, reward, complete = env.step(action_ind)
    return next_state, reward, complete


def norm_tensor(t):
    mean_, std_ = torch.mean(t), torch.std(t)
    return (t - mean_)/std_


"""
Global Variables:
"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure the environment
numOfRUs = 6  # Number of RUs. Number of Radio cards = numOfRus * 2
area_width = 500
area_height = 500
tx_power_levels = [30, 30]  # Example power levels
tx_frequencies = [60e9, 60e9]
numOfUEs = 1
# 2 x 2 x number of RUs are the total number of actions
action_space = range(2 * 2 * numOfRUs)
n_actions = len(action_space)
#n_observations = numOfUEs * (2 + numOfRUs * 2) # comment out if don't want locations
n_observations = numOfUEs * (numOfRUs * 2) 
# DQN
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
#memory = ReplayMemory(MAX_STEP * MAX_EPISODE)
memory = ReplayMemory(100 * MAX_EPISODE)

steps_done = 0  # to be used in decaying epsilon
avg_loss_episode = []
loss_step = []
avg_reward_episode, reward_per_step = [], []

def main():
    global numOfUEs
    global numOfRUs
    global n_observations
    global policy_net
    global target_net
    global steps_done
    global memory
    global optimizer
    global eps_threshold

    for n in [1, 5, 10,20, 30, 50, 75 , 100]:

        # =================initialization============================
        eps_threshold = 1
        avg_reward = []
        avg_loss = []
        step_loss = []
        steps_done = 0
        numOfUEs = n
        model_name = "model_" + str(numOfUEs)
        rewards_name = "rewards_" + str(numOfUEs) +".csv"
        avg_rewards_name = "avg_rewards_" + str(numOfUEs) +".csv"
        loss_name =  "loss_" + str(numOfUEs) +".csv"
        avg_loss_name = "avg_loss_" + str(numOfUEs) +".csv"
        n_observations = numOfUEs * (numOfRUs * 2)
        #memory = ReplayMemory(MAX_STEP * MAX_EPISODE)
        memory = ReplayMemory(100 * MAX_EPISODE)
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        
        # ===========================================================

        for episode in range(MAX_EPISODE):
            print("-")
            print(f"Episode: {episode}, Number of UEs: {numOfUEs}")
            env = Environment_scenario_0.Environment(
                numOfRUs, area_width, area_height, tx_power_levels, tx_frequencies, numOfUEs)
            play(env, avg_loss, step_loss, avg_reward, reward_per_step)
        plt.plot(avg_loss)
        
        plt.plot(avg_reward)
        
        avg_loss_episode.append(avg_loss)
        loss_step.append(step_loss)

        avg_reward_episode.append(avg_reward)


        torch.save(policy_net.state_dict(), model_name)
        
        # with open(loss_name, 'w', newline='') as file:
        #      writer = csv.writer(file)
        #      writer.writerow(loss_step)  # Write the list as a row in the CSV
        
        with open(avg_loss_name, 'w', newline='') as file:
             writer = csv.writer(file)
             writer.writerow(avg_loss)  # Write the list as a row in the CSV
             
        with open(avg_rewards_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(avg_reward)  # Write the list as a row in the CSV
             
             
        # test_net = DQN(n_observations, n_actions).to(device)
        # test_net.load_state_dict(torch.load("model_1"))
        # test_net.eval()
    with open('losses.pckl', 'wb') as f:
        pickle.dump([avg_loss_episode, loss_step], f)
        
    # f = open('losses.pckl', 'rb')
    # obj = pickle.load(f)
    # f.close()
    


if __name__ == "__main__":
    main()
