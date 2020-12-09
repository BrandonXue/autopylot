# Local modules
import inputs
from inputs import GameKeys1, QuitKeys
from learner import DeepQLearner
from pixelmaps import Map, MapLoader
from pixelworld import PixelWorld

# Non-local modules
import numpy as np
import pygame.time
from sys import argv

PLAY_FRAME_RATE = 4
TRAIN_FRAME_RATE = 120
RAW_WORLD_DIMENSIONS = (10, 10)

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
            done = pix.get_event_step(0)[2] # See docstring to see what numbers mean
        elif move == inputs.K_w:
            done = pix.get_event_step(1)[2]
        elif move == inputs.K_d:
            done = pix.get_event_step(2)[2]
        elif move == inputs.K_s:
            done = pix.get_event_step(3)[2]
        else:
            done = False
        # If game in terminal state, reset it
        if done:
            pix.reset()

def start_training(do_save: bool, game_map: Map) -> None:
    # TODO: Setup the game and start the training loop
    pix = PixelWorld(game_map)

    input_shape = game_map.dims + (3,)
    dqn = DeepQLearner(pix, input_shape)

    keys = QuitKeys()
    clock = pygame.time.Clock()

    pix.reset()
    while True:
        clock.tick(TRAIN_FRAME_RATE)    # Limit framerate for stability

        keys.grab_keys()                # Get inputs
        if keys.has_quit():             # See if game should quit
            break

        pix.get_event_step(-1)
        pix.render(show_data_surf=True)
        # print(np.array(pygame.surfarray.pixels3d(pix.data_surf)).T) # diagnostic only

def print_hint() -> None:
    print(
        'Usage:\n\n'
        f'python3 {argv[0]} -h | --help\n'
        f'python3 {argv[0]} train [--save] map_file\n'
        f'python3 {argv[0]} play map_file'
    )

def main():
    # Parse CLI args
    if argv[1] in {'train', 'play'}:
        op_mode = argv[1]
    elif argv[1] in {'-h', '--help'}:
        print_hint()
        return
    else:
        print('Unknown command.', end=' ')
        print_hint()
        return

    do_save = '--save' in argv

    if do_save:
        map_file = argv[3]
    else:
        map_file = argv[2]

    # Load a map based on filename from CLI args
    map_loader = MapLoader()
    try:
        game_map = map_loader.load(map_file)
    except Exception as e:
        print(e)
        return

    # Launch training mode
    if op_mode == 'train':
        start_training(do_save, game_map)
    
    # Launch play mode
    elif op_mode == 'play':
        start_game(game_map)

    return

if __name__ == '__main__':
    main()