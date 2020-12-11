# Local modules
import car_game
import learner
import filter_viz

# SESSION CONFIG
GAME_MODE = 'train'
MODEL_NAME = 'saves/learner_model'
TARGET_MODEL_NAME = 'saves/learner_target_model'
EXP_REPLAY_NAME = 'saves/learner_replay.data'
FLIP_DISPLAY = False
SAVE_TRAINING = True
MAP_NAME = 'maps/tight_corridors.txt'

def main():
    # Training mode with Keras
    if GAME_MODE == 'train':
        my_car_game = car_game.CarGame(GAME_MODE, FLIP_DISPLAY)
        my_car_game.load_map_from_file(MAP_NAME)
        my_car_game.reset()
        my_learner = learner.DQN(MODEL_NAME, TARGET_MODEL_NAME, EXP_REPLAY_NAME, SAVE_TRAINING)

        my_learner.set_game(my_car_game) # Set references
        my_car_game.set_learner(my_learner)

        my_learner.set_training_config()
        my_learner.load_or_create_models()
        my_learner.load_or_create_buffers()

        try:
            my_learner.run_q_model_test()
        except KeyboardInterrupt:
            my_learner.save_items()

    # Freeplay mode
    elif GAME_MODE == 'play':
        my_car_game = car_game.CarGame(GAME_MODE, FLIP_DISPLAY)
        my_car_game.load_map_from_file(MAP_NAME)
        my_car_game.reset()
        my_car_game.playing_game_loop()

    elif GAME_MODE =='viz':
        viz = filter_viz.Visualizer('saves/modeldata', TARGET_MODEL_NAME)
        if viz.loaded:
            viz.summarize()
            viz.show_filters(1)
            viz.show_filters(2)


if __name__ == '__main__':
    main()