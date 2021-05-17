from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, adjacent_positions, \
    row_col, translate, min_distance
from kaggle_environments import make

import gym
from gym import spaces

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum, auto
import numpy as np
import os
import random as rand


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
        self.last_min_distance_to_food = self.rows * self.columns  # initial max value to mark no food seen so far
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
        food = food if food != None else self.food
        return min_distance(position, food, self.columns)

    def _row_col(self, position):
        return row_col(position, self.columns)

    def _translate(self, position, direction):
        return translate(position, direction, self.columns, self.rows)

    def _preprocess_env(self, obs):
        observation = Observation(obs)

        self.my_index = observation.index

        if len(observation.geese[self.my_index]) > 0:
            self.my_head = observation.geese[self.my_index][0]
            self.my_tail = observation.geese[self.my_index][-1]
            self.my_body = [pos for pos in observation.geese[self.my_index][1:-1]]
        else:
            self.my_head = -1
            self.my_tail = -1
            self.my_body = []

        self.geese = [g for i, g in enumerate(observation.geese) if i != self.my_index and len(g) > 0]
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


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        """use names generated on adapted saved_dict
        dict_keys(['layer0.weight', 'layer0.bias', 'layer2.weight', 'layer2.bias', ...])

        net_arch as seen before:
          (q_net): QNetwork(
            (features_extractor): FlattenExtractor(
              (flatten): Flatten(start_dim=1, end_dim=-1)
            )
            (q_net): Sequential(
              (0): Linear(...)
              (1): ReLU()
              ...
            )
          )
        """
        self.even_layers = []
        net_arch = [77] + [100, 100, 300, 100, 100, 100, 100, 100] + [4]
        for inp, out in zip(net_arch[:-1], net_arch[1:]):
            self.even_layers.append(nn.Linear(inp, out))
        """
        #net_arch = [2000, 1000, 500, 1000, 500, 100]
        self.layer0 = nn.Linear(77, 2000)
        self.layer2 = nn.Linear(2000, 1000)
        self.layer4 = nn.Linear(1000, 500)
        self.layer6 = nn.Linear(500, 1000)
        self.layer8 = nn.Linear(1000, 500)
        self.layer10 = nn.Linear(500, 100)
        self.layer12 = nn.Linear(100, 4)
        """

    def forward(self, x):
        x = nn.Flatten()(x)  # no feature extractor means flatten (check policy arch on DQN creation)
        for layer in self.even_layers[:-1]:
            x = F.relu(layer(x))
        x = self.even_layers[-1](x)
        return x

        """
        for layer in [self.layer0, self.layer2, self.layer4, self.layer6, self.layer8, self.layer10]:
            x = F.relu(layer(x))
        x = self.layer12(x)
        """
        return x


def my_dqn(observation, configuration):
    global model, obs_prep, last_action, last_observation, previous_observation

    # tgz_agent_path = '/kaggle_simulations/agent/'
    # normal_agent_path = '/kaggle/working'
    tgz_agent_path = './'
    normal_agent_path = './'
    model_name = "dqnv_2opponents"
    num_previous_observations = 0
    epsilon = 0
    init = False
    debug = False

    try:
        model
    except NameError:
        init = True
    else:
        if model == None:
            init = True
            initializing
    if init:
        # initializations
        defaults = [configuration.rows,
                    configuration.columns,
                    configuration.hunger_rate,
                    configuration.min_food]

        model = MyNN()
        last_action = -1
        last_observation = []
        previous_observation = []

        file_name = os.path.join(normal_agent_path, f'{model_name}.pt')
        if not os.path.exists(file_name):
            file_name = os.path.join(tgz_agent_path, f'{model_name}.pt')

        model.load_state_dict(th.load(file_name), strict=False)
        obs_prep = ObservationProcessor(configuration.rows, configuration.columns, configuration.hunger_rate,
                                        configuration.min_food)

    # maintaint list of  last observations
    if num_previous_observations > 0 and len(last_observation) > 0:
        # Not initial step, t=0
        previous_observation.append(last_observation)
        # Keep list constrained to max length
        if len(previous_observation) > num_previous_observations:
            del previous_observation[0]

    # Convert to grid encoded with CellState values
    aux_observation = [obs_prep.process_env_obs(observation)]
    last_observation = aux_observation

    if num_previous_observations > 0 and len(previous_observation) == 0:
        # Initial step, t=0
        previous_observation = [last_observation for _ in range(num_previous_observations)]

    if num_previous_observations > 0:
        aux_observation = np.concatenate((*previous_observation, last_observation), axis=0)
    else:
        aux_observation = last_observation

    # predict with aux_observation.shape = (last_observations x rows x cols)
    tensor_obs = th.Tensor([aux_observation])
    n_out = model(tensor_obs)  # Example: tensor([[0.2742, 0.2653, 0.2301, 0.2303]], grad_fn=<SoftmaxBackward>)

    # choose probabilistic next move based on prediction outputs
    # with epsilon probability of fully random, always avoid opposite of last move
    actions = [action.value for action in Action]
    weights = list(n_out[0].detach().numpy())
    if last_action != -1:
        # Avoid dying by stupidity xD
        remove_index = actions.index(obs_prep.opposite(Action(last_action)).value)
        del actions[remove_index]
        del weights[remove_index]
    random = False

    min_value = abs(min(weights))
    weights = [min_value + w + 1e-5 for w in weights]  # Total of weights must be greater than zero

    # Reduce weight to penalize bad moves (collisions, etc...)
    weights_changed = False
    weights_before = [w for w in weights]
    for index, action in enumerate(actions):
        future_position = obs_prep._translate(obs_prep.my_head, Action(action))
        if not obs_prep.free_position(future_position):
            weights[index] = min(weights[index], 1e-8)  # Collision is worst case
            weights_changed = True
        elif future_position in obs_prep.dead_ends:
            weights[index] = min(weights[index], 1e-2)  # dead ends
            weights_changed = True
        elif future_position in obs_prep.adjacent_to_heads:
            weights[index] = min(weights[index], 1e-8)  # adjacent to heads
            weights_changed = True

    if debug and weights_changed:
        print(aux_observation)
        print(
            f'Adapted weights: before {weights_before} and after {weights} for actions {[Action(a).name for a in actions]}')
    # elif debug and not weights_changed:
    #    print(f'Action weights {weights}')

    if rand.random() < epsilon:
        prediction = rand.choice(actions)
        random = True
    else:
        prediction = rand.choices(actions, weights=weights)[0]
    action_predicted = Action(prediction).name

    # print(observation) #Uncomment to debug a bit too much...
    # if (last_action!=-1) and debug:
    #    print(last_observation)
    #    print(f'valid_actions={actions}, w={weights}, chose={Action(prediction).name}, rand={random}',
    #          f'previous={Action(last_action).name}, opposite={Action(obs_prep.opposite(Action(last_action)).value).name}')

    last_action = prediction
    return action_predicted  # return action
