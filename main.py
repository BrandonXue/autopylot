from car_game import CarGame
from data_processing import DataProcessor

from multiprocessing import Process, Pipe

def start_data_processing(game_to_data, data_to_game, game_mode, data_mode):
    # Keep the connections that data processor will use
    read_conn = game_to_data[0]
    write_conn = data_to_game[1]
    # Close other connections
    game_to_data[1].close()
    data_to_game[0].close()
    data_pro = DataProcessor(read_conn, write_conn, game_mode, data_mode)
    data_pro.start()

def start_game(game_to_data, data_to_game, game_mode, data_mode, no_flip):
    read_conn = data_to_game[0]
    write_conn = game_to_data[1]
    data_to_game[1].close()
    game_to_data[0].close()
    car_game = CarGame(read_conn, write_conn, game_mode, data_mode, no_flip)
    car_game.start()

def main():
    NO_FLIP = False # WARNING: only to be used on (train, pipe) mode
    GAME_MODE = 'train' # play or train
    DATA_MODE = 'pipe' # mss or pipe

    # Use two pipes because one process writing too quickly will fill up the pipe
    # read_end, write_end = Pipe(False)
    game_to_data = Pipe(duplex=False)
    data_to_game = Pipe(duplex=False)

    # Create a separate process for the game and for data processing
    data_proc = Process(
        target=start_data_processing,
        args=(game_to_data, data_to_game, GAME_MODE, DATA_MODE)
    )
    game_proc = Process(
        target=start_game,
        args=(game_to_data, data_to_game, GAME_MODE, DATA_MODE, NO_FLIP)
    )

    # Start both processes
    data_proc.start()
    game_proc.start()

    # Make sure this process isn't holding onto these file descriptors
    # This way when game quits, data will know that pipe closed/broke
    game_to_data[0].close()
    game_to_data[1].close()
    data_to_game[0].close()
    data_to_game[1].close()

    # Wait for processes to finish
    data_proc.join()
    game_proc.join()
    
        
if __name__ == '__main__':
    main()