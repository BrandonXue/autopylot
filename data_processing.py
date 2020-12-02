import sys
from numpy.core.getlimits import iinfo
from tensorflow.python.ops.gen_math_ops import mod
from config import *
from defs import ExitSignalType

from skimage import color, transform, exposure
import dill
import cv2
import mss
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DataProcessor:
    def __init__(self, read_conn, write_conn, game_mode, data_mode):
        self.game_mode = game_mode
        self.data_mode = data_mode

        self.read_conn = read_conn
        self.write_conn = write_conn

    def create_q_model(self):
        self.num_actions = 4
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(80, 80, 4,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(self.num_actions, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)


    def run_q_model_test(self):
        """Run the training. If mode is 'pipe', video data will be collected by piping
        from the car_game process. If mode is 'mss', video data will be collected by
        screen capture."""

        # Configuration for Keras setup
        # seed = 481             # For somewhat repeatable outcomes
        gamma = 0.99           # Discount factor for past rewards
        epsilon = 1.0          # Epsilon greedy parameter
        epsilon_min = 0.1      # Minimum for epsilon greedy parameter
        epsilon_max = 1.0      # Maximum epsilon greedy parameter
        epsilon_interval = (   # Rate at which to reduce chance of random action being taken
            epsilon_max - epsilon_min)  
        batch_size = 32        # Size of batch taken from replay buffer
        max_steps_per_episode = 10000


        # The first model makes the predictions for Q-values which are used to
        # make a action.
        model = self.create_q_model()

        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        model_target = self.create_q_model()

        # In the Deepmind paper they use RMSProp however then Adam optimizer
        # improves training time
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0
        # Number of frames to take random action and observe output
        epsilon_random_frames = 50000
        # Number of frames for exploration
        epsilon_greedy_frames = 1000000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        # Train the model after 6 actions
        update_after_actions = 6
        # How often to update the target network
        update_target_network = 10000
        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        while True:  # Run until solved

            received = self.get_event_step()

            pixels = received
            self.pipe_image(pixels)


            state = np.array(env.reset())
            episode_reward = 0

            for timestep in range(1, max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.num_actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, info_dict = env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1

            if running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

    def start(self):
        if self.data_mode == 'mss':
            self.setup_mss()

        if self.game_mode == 'play':
            self.start_play_mode()
        elif self.game_mode == 'train':
            self.start_train_mode()
        else:
            self.start_play_mode()


    def start_play_mode(self):
        while True:
            # Receive some data and process it a little
            state = self.get_event_step()

            # Send it back for HUD display
            self.pipe_image(state[0])

    def get_event_step(self):
        if self.data_mode == 'mss':
            return self.mss_event_step()
        elif self.data_mode == 'pipe':
            return self.pipe_event_step()
        else:# default pipe
            return self.pipe_event_step()

    def start_train_mode(self):
        dense = keras.layers.Dense(units=16)
        inputs = keras.Input(shape=(GRAYSCALE_DIM, GRAYSCALE_DIM, 3))
        
    def exit_procedure(self):  
        self.write_conn.send(ExitSignalType())
        # do some saving
        return

    def setup_mss(self):
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        
        self.monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": VIEWPORT_WIDTH,
            "height": VIEWPORT_HEIGHT,
            "mon": 1,
        }

    def mss_event_step(self):
        received = self.read_conn.recv()

        # If exit signal received, perform handshake and return
        if type(received) is ExitSignalType:
            self.exit_procedure()
            exit()
            
        pixels = np.array(self.sct.grab(self.monitor)).astype('float32')
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
        pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))# / 255
        #pixels = np.transpose(pixels.astype('float64'), (1,0,2))
        pixels = np.transpose(pixels, (1,0))

        # Observation, reward, done, info
        state = (pixels, received[1], received[2], None)
        return state
            

    def pipe_image(self, pixels):
        self.write_conn.send(pixels)

    def pipe_event_step(self):
        received = self.read_conn.recv()

        # If exit signal received, perform handshake and return
        if type(received) is ExitSignalType:
            self.exit_procedure()
            exit()
        
        pixels = received[0]
        pixels = np.array(pixels).astype('float32')
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))# / 255
        
        state = (pixels, received[1], received[2], None)
        return state
