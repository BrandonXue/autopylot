# Type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from car_game import CarGame

# Local modules
from config import *

# Other modules
import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

class DQN:
    def __init__(self, model_name: str, target_model_name: str, experience_replay_name: str, do_save: bool):
        self.model_name = model_name
        self.target_model_name = target_model_name
        self.experience_replay_name = experience_replay_name
        self.do_save = do_save

    def set_game(self, car_game_: 'CarGame') -> None:
        self.car_game = car_game_

    def create_q_model(self) -> keras.Model:
        # Treating the frames as channels
        # As long as the information is there, it should work
        inputs = layers.Input(shape=(55, 55, 4))

        # Convolution layers to detect features of the environment
        layer1 = layers.Conv2D(32, 7, strides=4, activation="relu")(inputs) # output 13x13
        layer2 = layers.Conv2D(64, 4, strides=3, activation="relu")(layer1) # output 4x4
        # Flatten before Dense layer
        layer3 = layers.Flatten()(layer2)
        # Dense layer to use feature maps to approximate Q-values
        layer4 = layers.Dense(512, activation="relu")(layer3)
        action = layers.Dense(self.num_actions, activation="tanh")(layer4)

        model = keras.Model(inputs=inputs, outputs=action)
        return model


    def set_training_config(self) -> None:
        self.num_actions = 6 # F, FL, FR, B, BL, BR
        self.solve_threshold = 10000
        self.gamma = 0.99           # Discount factor for future rewards
        self.epsilon = 1.0          # Epsilon for exploration
        self.epsilon_min = 0.1      # Minimum for epsilon greedy parameter
        self.epsilon_max = 1.0      # Maximum epsilon greedy parameter

        # Rate at which to reduce chance of random action being taken
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)  

        self.batch_size = 32        # Size of batch taken from replay buffer
        self.max_steps_per_episode = 10000

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 50000 # 50000 
        # Number of frames for exploration
        self.epsilon_greedy_frames = 1000000.0 # 1000000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        self.max_memory_length = 10000 # 10000
        # Train the model after 6 actions
        self.update_after_actions = 4
        # How often to update the target network
        self.update_target_network = 10000

    def load_or_create_models(self) -> None:
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        try:
            self.model = load_model(self.model_name)
        except:
            self.model = self.create_q_model()
            print('Model not loaded. Created new model')
        else:
            print('Model loaded successfully.')

        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        try:
            self.model_target = load_model(self.target_model_name)
        except:
            self.model_target = self.create_q_model()
            print('Model not loaded. Created new target model')
        else:
            print('Target model loaded successfully.')

    def load_or_create_buffers(self) -> None:
        # Experience replay buffers
        try:
            with open(self.experience_replay_name, 'rb') as filehandle:
                self.action_history = pickle.load(filehandle)
                self.state_history = pickle.load(filehandle)
                self.state_next_history = pickle.load(filehandle)
                self.rewards_history = pickle.load(filehandle)
                self.done_history = pickle.load(filehandle)
                self.episode_reward_history = pickle.load(filehandle)
        except:
            print('An error occured during loading of experience replay.')
            self.action_history = []
            self.state_history = []
            self.state_next_history = []
            self.rewards_history = []
            self.done_history = []
            self.episode_reward_history = []
        else:
            print('Experience replays loaded successfully')

    def add_to_buffers(self, action, state, state_next, done, reward) -> None:
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)

    def trim_buffers(self) -> None:
        del self.rewards_history[:1]
        del self.state_history[:1]
        del self.state_next_history[:1]
        del self.action_history[:1]
        del self.done_history[:1]

    # !!!!NOTE: Code from online: Keras Atari breakout example
    # https://keras.io/examples/rl/deep_q_network_breakout/
    # By authors: Jacob Chapman and Mathias Lechner
    def run_q_model_test(self) -> None:
        """Run the training. If mode is 'pipe', video data will be collected by piping
        from the car_game process. If mode is 'mss', video data will be collected by
        screen capture."""

        running_reward = 0
        episode_count = 0
        frame_count = 0

        # learning_rate 0.00025
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00026, clipnorm=1.0)
        
        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()

        while True:  # Run until solved
            self.car_game.reset()
            state = np.array(self.car_game.get_state())
            episode_reward = 0

            for timestep in range(1, self.max_steps_per_episode):
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    action = np.random.choice(self.num_actions) # Random action
                else:
                    # Predict action Q-values from environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                info = (running_reward, episode_count, frame_count)
                state_next, reward, done, info_dict = self.car_game.get_event_step(action, info)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                self.add_to_buffers(action, state, state_next, done, reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([self.state_history[i] for i in indices])
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    rewards_sample = [self.rewards_history[i] for i in indices]
                    action_sample = [self.action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = self.loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if frame_count % self.update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(self.rewards_history) > self.max_memory_length:
                    self.trim_buffers()

                if done:
                    break

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)

            episode_count += 1

            if running_reward > self.solve_threshold:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

        self.save_items()

        
    def save_items(self) -> None: 
        '''
        If self.do_save is set to False, nothing happens.
        Save the Model and experience replay buffers, if any.
        Exit the program. 
        ''' 

        # Check flag, do not save if saving is disabled
        if not self.do_save:
            print('Saving disabled. No saves were made!')
            return

        print('\n')
        try: # Save the model
            self.model.save(self.model_name)
        except:
            print('Model not saved.')
        else:
            print('Model saved successfully.')

        try: # Save the target model
            self.model_target.save(self.target_model_name)
        except:
            print('Target model not saved.')
        else:
            print('Target model saved successfully.')

        try: # Save experience replay buffers
            with open(self.experience_replay_name, 'wb') as filehandle:
                pickle.dump(self.action_history, filehandle)
                pickle.dump(self.state_history, filehandle)
                pickle.dump(self.state_next_history, filehandle)
                pickle.dump(self.rewards_history, filehandle)
                pickle.dump(self.done_history, filehandle)
                pickle.dump(self.episode_reward_history, filehandle)
        except:
            print('An error occured during saving of experience replay.')
        else:
            print('Experience replays saved successfully')
        print()