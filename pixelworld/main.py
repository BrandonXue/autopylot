# Local modules
import inputs
from inputs import GameKeys1, QuitKeys
from learner import DeepQLearner
from pixelmaps import Map, MapLoader
from pixelworld import PixelWorld


# Non-local modules
from random import random, randint
import tensorflow as tf
from tensorflow.python.keras.backend import argmax
import numpy as np
import pygame.time
from sys import argv

PLAY_FRAME_RATE = 4
TRAIN_FRAME_RATE = 200

def start_game(game_map: Map) -> None:
    # TODO: Setup the game and start the game loop
    pix = PixelWorld(game_map)
    pix.reset()
    keys = GameKeys1()
    clock = pygame.time.Clock()
    while True:
        clock.tick(PLAY_FRAME_RATE) # Limit framerate for playability

        keys.grab_keys()            # Get inputs
        if keys.has_quit():         # See if game should quit
            break

        pix.render()                # Render game world

        move = keys.get_move_key()  # See which movement to make (if any)
        # Advance game state
        if move == inputs.K_a:
            pix.perform_action(0)# See docstring to see what numbers mean
        elif move == inputs.K_w:
            pix.perform_action(1)
        elif move == inputs.K_d:
            pix.perform_action(2)
        elif move == inputs.K_s:
            pix.perform_action(3)

        # If game in terminal state, reset it
        if pix.get_done():
            pix.reset()

def start_training(do_save: bool, save_dir: str, do_load: bool, load_dir: str, game_map: Map) -> None:
    ''' 
    Please see Google DeepMind\'s paper: 
    https://arxiv.org/pdf/1312.5602v1.pdf
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    '''

    pix = PixelWorld(game_map)

    input_shape = game_map.dims + (3,)
    dqn = DeepQLearner(input_shape)
    if do_load:
        dqn.load(load_dir)
    else:
        dqn.set_config()
        dqn.create_models()
        dqn.create_optimizer()
        dqn.create_losses()
        dqn.create_replay_mem()

    if do_save: # Check if save path is somewhat valid before we run training
        dqn.try_save_path(save_dir)

    keys = QuitKeys()
    clock = pygame.time.Clock()

    config = dqn.config
    epsilon_dec = (config.epsilon_start - config.epsilon_min)/config.epsilon_anneal_frames
    epsilon = config.epsilon_start

    total_frames = 0
    for episode in range(config.max_episodes):
        # The deepmind authors used a phi function to preprocess sequences.
        # For pixelworld we don't have sequences so our preprocess
        state = pix.reset()
        state = dqn.preprocess(state)
        
        frame = 0
        while frame < config.max_episode_frames:
            frame += 1
            total_frames += config.k 

            clock.tick(TRAIN_FRAME_RATE)    # Limit framerate for stability

            keys.grab_keys()                # Get inputs
            if keys.has_quit():             # See if game should quit
                if do_save:
                    dqn.save(save_dir)
                exit()
            
            # With probability ε select a random action a_t
            if random() < epsilon:
                action = randint(0, pix.num_actions() - 1)
                
            # otherwise select a_t = max_a Q*( φ(s_t), a; θ)
            else:
                # Need to convert state into a state tensor with batch size 1 for prediction at specific state
                current_state_tf = tf.convert_to_tensor(state)
                # We want to go from (width, height, rgbchannels) to (1, width, height, rgbchannels) so axis 0
                current_state_tf = tf.expand_dims(current_state_tf, axis=0)

                print(current_state_tf.numpy().shape)
                # __call__  (i.e. dqn.model()) is equivalent to model.predict()
                # however it is recommended on smaller batch sizes to use __call__ 
                action_values = dqn.model(current_state_tf, training=False)
                # DeepMind mentions how architecture of one Q-value per feedforward is inefficient
                # Instead use architecture where Q-values of all actions are returned
                # Instead of feeding image and action into network, only feed image.
                action = np.argmax(action_values) # get the action with the highest Q-value

            # DeepMind's paper describes a frame-skipping technique.
            # Although this isn't applicable to our tiny pixel world (k = 1),
            # this training loop can be generalized to other games.

            # Update epsilon based on annealing schedule clamp to min
            epsilon = max(config.epsilon_min, epsilon - epsilon_dec) #this shoud change on every set of k frames

            # Get remaining part of sequence with phi (preprocessing).
            # Includes environment info (which is just to mimic the format of AIGym).
            # Execute action at in emulator and observe reward r_t and image x_{t+1}
            state_next, reward, done, info = pix.get_event_step(action, config.k) 
            state_next = dqn.preprocess(state_next)

            # Store transition in replay history
            dqn.add_transition(state, action, reward, state_next, done)

            # Update current state
            state = state_next

            #update model (per frame) target by sampling minibatch of size 32 (default for training)
            if (total_frames/config.k) > config.batch_size:
                (state_sample, action_sample, reward_sample, 
                state_next_sample, done_sample) = dqn.get_sample(config.batch_size)

                state_sample = tf.convert_to_tensor(state_sample)
                state_next_sample = tf.convert_to_tensor(state_next_sample)
                
                # Set y_j = r_j for terminal φ_{j+1}
                # Set y_j = r_j + γ max_{a'} Q( φ_{j+1}, a′; θ') for non-terminal φ_{j+1}

                # Use this to mask out the Q predictions for terminal states
                done_filter = [not done for done in done_sample]

                # Q( φ_{j+1}, a′; θ')
                q_pred_next = dqn.target_model.predict(state_next_sample, batch_size=config.batch_size)

                # max_{a'} Q( φ_{j+1}, a′; θ')
                max_q_next = tf.reduce_max( q_pred_next, axis=-1 )

                # For terminal states, set this Q value to 0. Also decay all Q values
                # γ max_{a'} Q( φ_{j+1}, a′; θ')
                discounted_future_q = tf.Variable(dqn.config.gamma * max_q_next * done_filter)

                # y_j
                target_y = reward_sample + discounted_future_q
                
                # Use this to mask out Q predictions for actions we didn't take
                mask = tf.one_hot(action_sample, pix.num_actions())

                with tf.GradientTape() as tape:
                    # Q*(a, s; θ)
                    q_pred = tf.reduce_max(tf.multiply(dqn.model(state_sample, training=True), mask), axis=-1)

                    loss = dqn.losses(target_y, q_pred)

                gradients = tape.gradient(loss, dqn.model.trainable_variables)
                dqn.opt.apply_gradients(zip(gradients, dqn.model.trainable_variables))

                print(loss)

            #after large number of frames update model
            # Authors describe updating model after C steps
            if (total_frames % config.C) == 0:
                dqn.target_model.set_weights(dqn.model.get_weights()) 
            pix.render(show_data_surf=True)

            if done:
                break

            # print(np.array(pygame.surfarray.pixels3d(pix.data_surf)).T) # diagnostic only

def print_hint() -> None:
    ''' Command line hints/help '''
    print(
        '\nUsage:\n'
        f"python3 {argv[0]} -h | --help\n"
        f"python3 {argv[0]} train map_file [--save save_dir] [--load load_dir]\n"
        f"python3 {argv[0]} play map_file\n\n"

        ".txt maps can have spaces or empty lines.\n"
        "empty = 0, pit = 1, pellet= 2, player spawn = 3\n\n"

        ".png maps should be in three channel colors.\n"
        "The folors are as follows (in RGB):\n"
        "empty = 0xFFFFFF, pit = 0x000000, pellet= 0x0000FF, player spawn = 0x00FF00"
    )

def main():
    argv_cp = argv # Make copy of CLI args
    
    # Get run mode
    if argv_cp[1] in {'train', 'play'}:
        op_mode = argv_cp[1]
        argv_cp.remove(op_mode)
    elif argv_cp[1] in {'-h', '--help'}:
        print_hint()
        return
    else:
        print('Unrecognized command. Please check your spelling.', end=' ')
        print_hint()
        return

    # Get map
    if len(argv_cp) > 1:
        map_file = argv_cp[1]
        argv_cp.remove(map_file)
    else:
        print('Please specify map_file second.', end=' ')
        print_hint()
        return
    
    # Get save flags
    try:
        save_flag = argv_cp.index('--save')
        do_save = True
        save_dir = argv_cp[save_flag+1]
        argv_cp.pop(save_flag+1)
        argv_cp.remove('--save')
    except:
        do_save = False
        save_dir = ''

    # Get load flags
    try:
        load_flag = argv_cp.index('--load')
        do_load = True
        load_dir = argv_cp[load_flag+1]
        argv_cp.pop(load_flag+1)
        argv_cp.remove('--load')
    except:
        do_load = False
        load_dir = ''
        
    if len(argv_cp) > 1:
        print(f"Unknown arguments: {argv_cp[1:]}")
        print_hint()
        return

    # Load a map based on filename from CLI args
    try:
        game_map = MapLoader.load(map_file)
    except Exception as e:
        print(e)
        return

    # Launch training mode
    if op_mode == 'train':
        start_training(do_save, save_dir, do_load, load_dir, game_map)
    
    # Launch play mode
    elif op_mode == 'play':
        start_game(game_map)

    return

if __name__ == '__main__':
    main()