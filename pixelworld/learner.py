# Authors:
# Brandon Xue       brandonx@csu.fullerton.edu
# Jacob Rapmund     jacobwrap86@csu.fullerton.edu
#
# This module contains functionality for generating a
# deep reinforcement learning agent.
# The neural network models are defined here.
# Functionality for serializing stateful components are
# also in this module.

#  Local modules
import utils

# Non-local modules
from collections import namedtuple
import numpy as np
import os
import pathlib
import pickle
from tensorflow import keras
from tensorflow.keras import layers, models

DQLConfig = namedtuple(
    'DQLConfig',
    [
        # Maximum number of items to store in experience memory.
        'mem_len',

        # Maximum frames per episode.
        'max_episode_frames',

        # Maximum episodes to train for.
        'max_episodes',

        # Minimum value for epsilon. After annealing ends epsilon stays at this value.
        'epsilon_min',

        # Starting value for epsilon, the probability a random action
        # is taken over the greedy action given by Q.
        'epsilon_start',

        # Discount factor for rewards, R_t = r_t + gamma * Q(s', a')
        'gamma',

        # Over how many frames to linearly reduce epsilon.
        # DeepMind reduced over 1M frames, 1/10 of total training frames
        'epsilon_anneal_frames',

        # Defines frame skip. The same action is taken for k frames.
        # The agent sees every k frames.                    
        'k',

        # Defines the size of a mini-batch takes from experience replay memory.
        # The authors of DeepMind used 32.
        'batch_size',

        # Defines the frequency (once per this many frames)
        # it takes for a target model update to occur
        'C',
    ]
)

class DeepQLearner:
    def __init__(self, input_shape):
        self.input_shape = input_shape

        # Config
        self.config = None

        # Keras Models
        self.model = None
        self.target_model = None

        # Optimizer
        self.opt = None

        # Loss Function
        self.losses = None

        # Replay Memory
        self.state_mem = None
        self.action_mem = None
        self.reward_mem = None
        self.state_next_mem = None
        self.done_mem = None

    def preprocess(self, image, dims=(9, 9)) -> np.ndarray:
        # Convert to numpy ndarray
        image = np.array(image)

        # Crop to 9x9 for those larger maps
        width, height, channels = image.shape
        x_margin = max((width - 9), 0) // 2     # Half the excess width, always non-negative
        y_margin = max((height - 9), 0) // 2    # Half the excess height, always non-negative

        # If image is smaller than 9x9, it will not be padded and will remain small in the output
        return image[x_margin:x_margin+dims[0], y_margin:y_margin+dims[1], :]


    def get_sample(self, batch_size: int):
        all_indices = range(len(self.state_mem))
        selections = np.random.choice(all_indices, batch_size, replace=False)

        return (
            [self.state_mem[i] for i in selections],
            [self.action_mem[i] for i in selections],
            [self.reward_mem[i] for i in selections],
            [self.state_next_mem[i] for i in selections],
            [self.done_mem[i] for i in selections]
        )

    def add_transition(self, state, action, reward, state_next, done):
        '''
        Add a state transition to the experience memory replay.

        Limits memory to mem_len defined in config.
        '''

        self.state_mem.append(state)
        if len(self.state_mem) > self.config.mem_len:
            del self.state_mem[:1]

        self.action_mem.append(action)
        if len(self.action_mem) > self.config.mem_len:
            del self.action_mem[:1]

        self.reward_mem.append(float(reward))
        if len(self.reward_mem) > self.config.mem_len:
            del self.reward_mem[:1]

        self.state_next_mem.append(state_next)
        if len(self.state_next_mem) > self.config.mem_len:
            del self.state_next_mem[:1]

        self.done_mem.append(done)
        if len(self.done_mem) > self.config.mem_len:
            del self.done_mem[:1]

    def set_config(self):
        self.config = DQLConfig(
            mem_len=100000,
            max_episode_frames=10000,
            max_episodes=100000,
            epsilon_start=1.0,
            epsilon_min=0.1,
            gamma=0.99,
            epsilon_anneal_frames=1000000,
            k=1,
            batch_size=32,
            C=1000
        )

    def load(self, path_to_dir: str) -> None:
        ''' 
        Load all training objects from a subdirectory within the saves dir.

        Specify a path to the subdirectory (including the subdirectory itself).

        e.g. ./saves/previous_save

        If part of this learner was already initialized, raises an Exception.
        If the target subdirectory is not found, raises an Exception.
        '''

        # Make sure all items to be saved exist
        if self.config != None: # Config tuple
            raise Exception("Config already initialized.")

        if self.model != None or self.target_model != None: # Keras Models
            raise Exception("One or more models already initialized.")

        if self.opt != None: # Optimizer
            raise Exception("Optimizer already initialized.")

        if self.losses != None: # Loss Function
            raise Exception("Loss function already initialized.")

        if type(None) != type(self.state_mem): # Replay Memory
            raise Exception("Experience replay memory already initialized.")

        # Make sure directory is correct
        if not os.path.exists(path_to_dir):
            raise Exception("Could not find the target subdirectory.")
        path_to_dir = utils.add_trailing_slash(path_to_dir)

        DeepQLearner.__load_helper(self, path_to_dir) # Everything ok, load

    @staticmethod
    def __load_helper(learner: 'DeepQLearner', path_to_dir: str):
        ''' To be called only when subdirectory has been created and is the cwd.'''

        with open(path_to_dir + 'config.dat', 'rb') as config_file:
            learner.config = pickle.load(config_file)

        learner.model = models.load_model(path_to_dir + 'model', compile=False)
        learner.target_model = models.load_model(path_to_dir + 'target_model', compile=False)
        # with open('models.dat', 'rb') as model_file:
        #     learner.model = pickle.load(model_file)
        #     learner.target_model = pickle.load(model_file)
            
        with open(path_to_dir + 'opt.dat', 'rb') as opt_file:
            learner.opt = pickle.load(opt_file)

        with open(path_to_dir + 'loss.dat', 'rb') as loss_file:
            learner.losses = pickle.load(loss_file)

        with open(path_to_dir + 'replay.dat', 'rb') as replay_file:
            data = pickle.load(replay_file)
            learner.state_mem = data['State Memory']
            learner.action_mem = data['Action Memory']
            learner.reward_mem = data['Reward Memory']
            learner.state_next_mem = data['State Next Memory']
            learner.done_mem = data['Done Memory']

    @staticmethod
    def try_save_path(path_to_dir: str) -> bool:
        ''' 
        To be called before a long training session so the save doesn't
        fail at the very end by surprise.

        May raise Exception if path is invalid.
        '''

        parent_path, child_dir = utils.find_parent_child(path_to_dir)
        if not os.path.exists(parent_path):
            return False
        elif not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        return True

    def save(self, path_to_dir: str) -> None:
        ''' 
        Save all training objects into a subdirectory within the saves dir.

        Specify a path to the subdirectory (including the subdirectory itself).

        e.g. ./saves/new_save

        If part of this learner was not initialized, raises an Exception.
        If the saves directory is not found, raises an Exception.
        '''

        # Make sure all items to be saved exist
        if self.config == None: # Config tuple
            raise Exception("Config not initialized.")

        if self.model == None or self.target_model == None: # Keras Models
            raise Exception("One or more models not initialized.")

        if self.opt == None: # Optimizer
            raise Exception("Optimizer not initialized.")

        if self.losses == None: # Loss Function
            raise Exception("Loss function not initialized.")

        if type(None) == type(self.state_mem): # Replay Memory
            raise Exception("Experience replay memory not initialized.")

        parent_path, child_dir = utils.find_parent_child(path_to_dir)
        if not os.path.exists(parent_path):
            raise Exception("Could not find destination path for saving.")

        elif not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)

        path_to_dir = utils.add_trailing_slash(path_to_dir)

        DeepQLearner.__save_helper(self, path_to_dir) # Everything okay, save


    @staticmethod
    def __save_helper(learner: 'DeepQLearner', path_to_dir: str):
        ''' To be called only when subdirectory has been created and is the cwd.'''

        with open(path_to_dir + 'config.dat', 'wb') as config_file:
            pickle.dump(learner.config, config_file)

        learner.model.save(path_to_dir + 'model', include_optimizer=False)
        learner.target_model.save(path_to_dir + 'target_model', include_optimizer=False)
            
        with open(path_to_dir + 'opt.dat', 'wb') as opt_file:
            pickle.dump(learner.opt, opt_file)

        with open(path_to_dir + 'loss.dat', 'wb') as loss_file:
            pickle.dump(learner.losses, loss_file)

        data = {
            'State Memory': learner.state_mem,
            'Action Memory': learner.action_mem,
            'Reward Memory': learner.reward_mem,
            'State Next Memory': learner.state_next_mem,
            'Done Memory': learner.done_mem
        }

        with open(path_to_dir + 'replay.dat', 'wb') as replay_file:
            pickle.dump(data, replay_file)

    def create_models(self):

        # Two models are used:
        # "The target net- work parameters are only updated with the Q-network parameters
        # every C steps and are held fixed between individual updates"
        self.model = self.__create_model()
        self.target_model = self.__create_model()

    def create_replay_mem(self):
        self.state_mem = []
        self.action_mem = []
        self.reward_mem = []
        self.state_next_mem = []
        self.done_mem = []

    def create_optimizer(self):
        ''' Create optimizer algorithm. '''

        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        # Nesterov-accelerated Adaptive Moment Estimation
        optimizer = keras.optimizers.Nadam(
            learning_rate=0.001,    # default 0.001    eta
            beta_1=0.99,    # default 0.9    exponential decay rate for momentum / momentum schedule
            beta_2=0.999,   # default 0.999  exponential decary rate  for the exponentially weighted infinity norm
            epsilon=1e-8    # default 1e-7   fuzzing to prevent divide by zero
                            # not to be confused with the greedy epsilon used for random actions
        )

        # DeepMind used RMSProp, but Adam seems to be acceptable as well.

        self.opt = optimizer

    def create_losses(self):
        '''
        Although the loss function is stateless, we serialize it so that
        future training sessions are using the same loss function and
        we don't accidentally change the loss function in between sessions. '''

        # DeepMind seems to be using MSE. But other example used Huber. 
        self.losses = keras.losses.MeanSquaredError()

    def __create_model(self):
        ''' Create our Deep Q Network based on DeepMind's paper. '''

        model = keras.Sequential([
            layers.Conv2D(
                32,  # number of filters / depth of output feature map
                5,  # kernel size / size of convolutional matrix
                strides=(1, 1), # how much kernel slides over each convolution
                activation='relu', # DeepMind describes using ReLU for their convolutional layers
                input_shape=self.input_shape,   # specified in constructor
                data_format='channels_last'     # (w, h, channels)
            ),
            layers.Conv2D(
                64, # number of filters / depth of output feature map
                3,  # kernel size / size of convolutional matrix
                strides=(1, 1), activation='relu',
                data_format='channels_last'
            ),
            layers.Flatten(data_format='channels_last'), # Flatten before Dense
            layers.Dense(
                units=256,
                activation='relu'
            ),
            layers.Dense(
                units=4,    # Use architecture with one output per action
                activation='linear' # DeepMind describes using linear for output layer
            )
        ])
        return model