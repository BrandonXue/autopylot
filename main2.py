import car_game2
import learner

def main():
    # SESSION CONFIG
    GAME_MODE = 'train'
    MODEL_NAME = 'saves/learner_model'
    TARGET_MODEL_NAME = 'saves/learner_target_model'
    EXP_REPLAY_NAME = 'saves/learner_replay.data'



    car_game = car_game2.CarGame(GAME_MODE, True)
    car_game.load_map_from_file("maps/game_map_1.txt")
    car_game.reset_map()

    # Training mode with Keras
    if GAME_MODE == 'train':
        my_learner = learner.Learner(MODEL_NAME, TARGET_MODEL_NAME, EXP_REPLAY_NAME)

        my_learner.set_game(car_game) # Set references
        car_game.set_learner(my_learner)

        my_learner.set_hyper_params()
        my_learner.load_or_create_models()
        my_learner.load_or_create_buffers()

        my_learner.run_q_model_test()

    # Freeplay mode
    elif GAME_MODE == 'play':
        car_game.playing_game_loop()

if __name__ == '__main__':
    main()