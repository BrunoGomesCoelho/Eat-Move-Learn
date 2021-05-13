
import gym
from gym import spaces

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, adjacent_positions, row_col, translate, min_distance
from kaggle_environments import make

from enum import Enum, auto
import numpy as np


class CellState(Enum):
    EMPTY = 0
    FOOD = auto()
    HEAD = auto()
    BODY = auto()
    TAIL = auto()
    MY_HEAD = auto()
    MY_BODY = auto()
    MY_TAIL = auto()
    ANY_GOOSE = auto()


class ObservationProcessor:

    def __init__(self, rows, columns, hunger_rate, min_food, debug=False, center_head=True):
        self.debug = debug
        self.rows, self.columns = rows, columns
        self.hunger_rate = hunger_rate
        self.min_food = min_food
        self.previous_action = -1
        self.last_action = -1
        self.last_min_distance_to_food = self.rows *self.columns  # initial max value to mark no food seen so far
        self.center_head = center_head

    # ***** BEGIN: utility functions ******
    def opposite(self, action):
        if action == Action.NORTH:
            return Action.SOUTH
        if action == Action.SOUTH:
            return Action.NORTH
        if action == Action.EAST:
            return Action.WEST
        if action == Action.WEST:
            return Action.EAST
        raise TypeError(str(action) + " is not a valid Action.")

    def _adjacent_positions(self, position):
        return adjacent_positions(position, self.columns, self.rows)

    def _min_distance_to_food(self, position, food=None):
        food = food if food !=None else self.food
        return min_distance(position, food, self.columns)

    def _row_col(self, position):
        return row_col(position, self.columns)

    def _translate(self, position, direction):
        return translate(position, direction, self.columns, self.rows)

    def _preprocess_env(self, obs):
        observation = Observation(obs)

        self.my_index = observation.index

        if len (observation.geese[self.my_index] ) >0:
            self.my_head = observation.geese[self.my_index][0]
            self.my_tail = observation.geese[self.my_index][-1]
            self.my_body = [pos for pos in observation.geese[self.my_index][1:-1]]
        else:
            self.my_head = -1
            self.my_tail = -1
            self.my_body = []


        self.geese = [g for i ,g in enumerate(observation.geese) if i!= self.my_index and len(g) > 0]
        self.geese_cells = [pos for g in self.geese for pos in g if len(g) > 0]

        self.occupied = [p for p in self.geese_cells]
        self.occupied.extend([p for p in observation.geese[self.my_index]])

        self.heads = [g[0] for i, g in enumerate(observation.geese) if i != self.my_index and len(g) > 0]
        self.bodies = [pos for i, g in enumerate(observation.geese) for pos in g[1:-1] if
                       i != self.my_index and len(g) > 2]
        self.tails = [g[-1] for i, g in enumerate(observation.geese) if i != self.my_index and len(g) > 1]
        self.food = [f for f in observation.food]

        self.adjacent_to_heads = [pos for head in self.heads for pos in self._adjacent_positions(head)]
        self.adjacent_to_bodies = [pos for body in self.bodies for pos in self._adjacent_positions(body)]
        self.adjacent_to_tails = [pos for tail in self.tails for pos in self._adjacent_positions(tail)]
        self.adjacent_to_geese = self.adjacent_to_heads + self.adjacent_to_bodies
        self.danger_zone = self.adjacent_to_geese

        # Cell occupation
        self.cell_states = [CellState.EMPTY.value for _ in range(self.rows * self.columns)]
        for g in self.geese:
            for pos in g:
                self.cell_states[pos] = CellState.ANY_GOOSE.value
        for pos in self.heads:
            self.cell_states[pos] = CellState.ANY_GOOSE.value
        for pos in self.my_body:
            self.cell_states[pos] = CellState.ANY_GOOSE.value
        self.cell_states[self.my_tail] = CellState.ANY_GOOSE.value

        # detect dead-ends
        self.dead_ends = []
        for pos_i, _ in enumerate(self.cell_states):
            if self.cell_states[pos_i] != CellState.EMPTY.value:
                continue
            adjacent = self._adjacent_positions(pos_i)
            adjacent_states = [self.cell_states[adj_pos] for adj_pos in adjacent if adj_pos != self.my_head]
            num_blocked = sum(adjacent_states)
            if num_blocked >= (CellState.ANY_GOOSE.value * 3):
                self.dead_ends.append(pos_i)

        # check for extended dead-ends
        new_dead_ends = [pos for pos in self.dead_ends]
        while new_dead_ends != []:
            for pos in new_dead_ends:
                self.cell_states[pos] = CellState.ANY_GOOSE.value
                self.dead_ends.append(pos)

            new_dead_ends = []
            for pos_i, _ in enumerate(self.cell_states):
                if self.cell_states[pos_i] != CellState.EMPTY.value:
                    continue
                adjacent = self._adjacent_positions(pos_i)
                adjacent_states = [self.cell_states[adj_pos] for adj_pos in adjacent if adj_pos != self.my_head]
                num_blocked = sum(adjacent_states)
                if num_blocked >= (CellState.ANY_GOOSE.value * 3):
                    new_dead_ends.append(pos_i)

    def safe_position(self, future_position):
        return (future_position not in self.occupied) and (future_position not in self.adjacent_to_heads) and (
                    future_position not in self.dead_ends)

    def valid_position(self, future_position):
        return (future_position not in self.occupied) and (future_position not in self.dead_ends)

    def free_position(self, future_position):
        return (future_position not in self.occupied)

        # ***** END: utility functions ******

    def process_env_obs(self, obs):
        self._preprocess_env(obs)

        EMPTY = .4
        HEAD = -1
        BODY = MY_BODY = -.8
        TAIL = MY_TAIL = -.5
        MY_HEAD = 0
        FOOD = 1
        RISK = -.5

        # Example: {'remainingOverageTime': 12, 'step': 0, 'geese': [[62], [50]], 'food': [7, 71], 'index': 0}
        # observation = [[CellState.EMPTY.value for _ in range(self.columns)] for _ in range(self.rows)]
        observation = [[EMPTY for _ in range(self.columns)] for _ in range(self.rows)]

        # Other agents
        for pos in self.heads:
            r, c = self._row_col(pos)
            observation[r][c] = HEAD  # CellState.HEAD.value
        for pos in self.bodies:
            r, c = self._row_col(pos)
            observation[r][c] = BODY  # CellState.BODY.value
        for pos in self.tails:
            r, c = self._row_col(pos)
            observation[r][c] = TAIL  # CellState.TAIL.value

        # Me
        r, c = self._row_col(self.my_head)
        observation[r][c] = MY_HEAD  # -1 #CellState.MY_HEAD.value
        if self.my_head != self.my_tail:
            r, c = self._row_col(self.my_tail)
            observation[r][c] = MY_TAIL  # CellState.MY_TAIL.value
        for pos in self.my_body:
            r, c = self._row_col(pos)
            observation[r][c] = MY_BODY  # CellState.MY_BODY.value

        # Food
        for pos in self.food:
            r, c = self._row_col(pos)
            observation[r][c] = FOOD  # CellState.FOOD.value

        if (self.previous_action != -1):
            aux_previous_pos = self._translate(self.my_head, self.opposite(self.previous_action))
            r, c = self._row_col(aux_previous_pos)
            if observation[r][c] > 0:
                observation[r][c] = MY_BODY * .5  # Marked to avoid opposite moves

        # Add risk mark
        for pos in self.adjacent_to_heads:
            r, c = self._row_col(pos)
            if observation[r][c] > 0:
                observation[r][c] = RISK

        # Add risk mark
        for pos in self.dead_ends:
            r, c = self._row_col(pos)
            if observation[r][c] > 0:
                observation[r][c] = RISK / 2

        if self.center_head:
            # NOTE: assumes odd number of rows and columns
            head_row, head_col = self._row_col(self.my_head)
            v_center = (self.columns // 2)  # col 5 on 0-10 (11 columns)
            v_roll = v_center - head_col
            h_center = (self.rows // 2)  # row 3 on 0-7 (7 rows)
            h_roll = h_center - head_row
            observation = np.roll(observation, v_roll, axis=1)
            observation = np.roll(observation, h_roll, axis=0)

        return np.array([observation])

    def common_sense_rewards(self, action):
        if self.my_head == -1:
            if self.debug:
                print("DIED!!")
            return -2

        reward = 0
        future_position = self._translate(self.my_head, action)
        check_opposite = (self.previous_action != -1)

        if future_position in self.occupied:
            if self.debug:
                print("Move to occupied")
            reward = -2  # this action meant death
        elif check_opposite and (self.previous_action == self.opposite(
                action)):  # opposite is currently a patch until Action.opposite works...
            if self.debug:
                print("Move to opposite direction, previous", self.previous_action, "vs now", action)
            reward = -2  # this action meant death
        elif (future_position in self.food) and (future_position not in self.adjacent_to_heads):
            if self.debug:
                print("Safe move to EAT!")
            reward = 2  # eating is good!
        elif future_position in self.dead_ends:
            if self.debug:
                print("Move to dead end")
            reward = 0
        else:
            min_distance_to_food = self._min_distance_to_food(future_position)

            if min_distance_to_food <= self.last_min_distance_to_food:
                if self.debug:
                    print("Move to food")
                # Removed positive rewards here, eating reward will be considered via gamma (future rewards) if agent gets to food
                if future_position in self.danger_zone:
                    reward = 0  # 0.1
                else:
                    reward = 0  # 0.2
            else:
                # ignore might be moving away, but also the nearest food could have been eaten... NO PENALTY HERE!
                reward = 0

            self.last_min_distance_to_food = min_distance_to_food

        self.previous_action = self.last_action
        self.last_action = action
        return reward


# Initial template from: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
class HungryGeeseEnv(gym.Env):

    def __init__(self, dummy_env=False, opponent=['greedy', 'greedy', 'greedy-goose.py'], action_offset=1, debug=False,
                 defaults=[7, 11, 10, 2]):
        super(HungryGeeseEnv, self).__init__()
        self.num_envs = 1
        self.num_previous_observations = 0
        self.debug = debug
        self.actions = [action for action in Action]
        self.action_offset = action_offset
        if not dummy_env:
            self.env = make("hungry_geese", debug=self.debug)
            self.rows = self.env.configuration.rows
            self.columns = self.env.configuration.columns
            self.hunger_rate = self.env.configuration.hunger_rate
            self.min_food = self.env.configuration.min_food
            self.trainer = self.env.train([None, *opponent])
        else:
            self.env = None
            self.rows = defaults[0]
            self.columns = defaults[1]
            self.hunger_rate = defaults[2]
            self.min_food = defaults[3]

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.num_previous_observations + 1, self.rows, self.columns),
                                            dtype=np.int8)
        self.reward_range = (-4, 1)
        self.step_num = 1
        self.observation_preprocessor = ObservationProcessor(self.rows, self.columns, self.hunger_rate, self.min_food,
                                                             debug=self.debug, center_head=True)
        self.observation = []
        self.previous_observation = []

    def step(self, action):
        action += self.action_offset
        action = Action(action)
        cs_rewards = self.observation_preprocessor.common_sense_rewards(action)
        if self.debug:
            if cs_rewards != 0:
                print("CS reward", action.name, self.observation, cs_rewards)
            else:
                print("CS ok", action.name)

        obs, reward, done, _ = self.trainer.step(action.name)

        if len(self.observation) > 0:
            # Not initial step, t=0
            self.previous_observation.append(self.observation)
            # Keep list constrained to max length
            if len(self.previous_observation) > self.num_previous_observations:
                del self.previous_observation[0]

        self.observation = self.observation_preprocessor.process_env_obs(obs)

        if len(self.previous_observation) == 0:
            # Initial step, t=0
            self.previous_observation = [self.observation for _ in range(self.num_previous_observations)]

        info = {}
        # if self.debug:
        #    print(action, reward, cs_rewards, done, "\n"+"\n".join([str(o) for o in self.observation]))

        env_reward = reward
        if len(self.previous_observation) > 0:
            unique_before, counts_before = np.unique(self.previous_observation[-1], return_counts=True)
            unique_now, counts_now = np.unique(self.observation, return_counts=True)
            before = dict(zip(unique_before, counts_before))
            now = dict(zip(unique_now, counts_now))
            count_length = lambda d: d.get(CellState.MY_HEAD.value, 0) + d.get(CellState.MY_BODY.value, 0) + d.get(
                CellState.MY_TAIL.value, 0)
            if count_length(now) > count_length(before):
                reward = 2  # Ate
            else:
                reward = 0  # Just moving
            if self.debug:
                print(f'{self.step_num} {count_length(now)} {count_length(before)} R {reward}')
        else:
            reward = 0  # no way to check previuos length use common sense reward on move to food instead ;-)
        if done:
            # game ended
            if self.observation_preprocessor.my_head == -1:
                # DIED, but what final ranking?
                rank = len(self.observation_preprocessor.geese) + 1
                if self.debug:
                    print("Rank on end", rank, "geese", self.observation_preprocessor.geese)
                if rank == 4:
                    reward = -2
                elif rank == 3:
                    reward = 0
                elif rank == 2:
                    reward = 0
                else:
                    reward = 100
            else:
                reward = 1  # survived the game!?
        elif reward < 1:
            reward = 1.1  # 0 #set to 0 if staying alive is not enough
        elif reward > 1:
            # ate something!!! :-)
            reward = 1

        if self.debug and done:
            print("DONE!", self.observation, env_reward, reward, cs_rewards)

        reward = cs_rewards if cs_rewards < 0 else cs_rewards + reward  # if cs_reward<0 use it only
        self.step_num += 1

        if self.num_previous_observations > 0:
            observations = np.concatenate((*self.previous_observation, self.observation), axis=0)
            return observations, reward, done, info
        else:
            return self.observation, reward, done, info

    def reset(self):
        self.observation_preprocessor = ObservationProcessor(self.rows, self.columns, self.hunger_rate, self.min_food,
                                                             debug=self.debug, center_head=True)
        obs = self.trainer.reset()
        self.observation = self.observation_preprocessor.process_env_obs(obs)
        self.previous_observation = [self.observation for _ in range(self.num_previous_observations)]
        return self.observation

    def render(self, **kwargs):
        self.env.render(**kwargs)


# In[ ]:


env = HungryGeeseEnv(opponent=['greedy', 'greedy', 'greedy'], debug=False)

from stable_baselines3.common.env_checker import check_env

check_env(env)

# In[ ]:


from stable_baselines3 import DQN

# In[ ]:


from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import torch.nn.functional as F

model_name = "dqnv_3oppponents_2ndtry"
m_env = Monitor(env, model_name, allow_early_resets=True)

policy_kwargs = dict(
    # net_arch = [2000, 1000, 500, 1000, 500, 100],
    net_arch=[100, 100, 300, 100, 100, 100, 100, 100],
    activation_fn=th.nn.ReLU
)

TRAIN_STEPS = 1e6
alpha_0 = 1e-6
alpha_end = 1e-9


def learning_rate_f(process_remaining):
    # default =  1e-4
    initial = alpha_0
    final = alpha_end
    interval = initial - final
    return final + interval * process_remaining


params = {
    'gamma': .9,
    'batch_size': 100,
    # 'train_freq': 500,
    'target_update_interval': 10000,
    'learning_rate': learning_rate_f,
    'learning_starts': 1000,
    'exploration_fraction': .2,
    'exploration_initial_eps': .05,
    'tau': 1,
    'exploration_final_eps': .01,
    'buffer_size': 100000,
    'verbose': 2,
}

# coment **params for default parameters
trainer = DQN('MlpPolicy', m_env, policy_kwargs=policy_kwargs, **params)

# You can check policy architecture with:
# print(trainer.policy.net_arch) #prints: [64, 64] for default DQN policy
# Or check model.policy
print(trainer.policy)

# In[ ]:


trainer.learn(total_timesteps=10000000, callback=None)

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv(f'{model_name}.monitor.csv', header=1, index_col='t')

df.rename(columns={'r': 'Episode Reward', 'l': 'Episode Length'}, inplace=True)
plt.figure(figsize=(20, 5))
sns.regplot(data=df, y='Episode Reward', x=np.arange(len(df)))

# In[ ]:


state_dict = trainer.policy.state_dict()
print("\n".join(state_dict.keys()))  # use this to check keys ;-)

# In[ ]:


adapted_state_dict = {
    new_key: state_dict[old_key]
    for old_key in state_dict.keys()
    for new_key in ["layer" + ".".join(old_key.split(".")[-2:])]  # use last 3 components of name
    if old_key.find("q_net_target.") != -1  # we only want the policy weights
}
print(adapted_state_dict.keys())
th.save(adapted_state_dict, f'{model_name}.pt')

# In[ ]:


state_dict.keys()

# In[ ]:


adapted_state_dict.keys()

# In[ ]:







