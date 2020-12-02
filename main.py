from car_game import CarGame
from data_processing import DataProcessor

from multiprocessing import Process, Pipe

def start_data_processing(pipe_connection, exit_signal, game_mode, data_mode):
    data_pro = DataProcessor(pipe_connection, exit_signal, game_mode, data_mode)
    data_pro.start()

def start_game(pipe_connection, exit_signal, game_mode, data_mode):
    car_game = CarGame(pipe_connection, exit_signal, game_mode, data_mode)
    car_game.start()

def main():
    GAME_MODE = 'play'
    DATA_MODE = 'mss'

    # Use two pipes because one process writing too quickly will fill up the pipe
    # read_end, write_end = Pipe(False)
    data_read_end, game_write_end = Pipe(duplex=False)
    game_read_end, data_write_end = Pipe(duplex=False)

    # Create a separate process for the game and for data processing
    data_proc = Process(
        target=start_data_processing,
        args=(data_read_end, data_write_end, GAME_MODE, DATA_MODE)
    )
    game_proc = Process(
        target=start_game,
        args=(game_read_end, game_write_end, GAME_MODE, DATA_MODE)
    )

    # Start both processes
    data_proc.start()
    game_proc.start()

    # Wait for processes to finish
    data_proc.join()
    game_proc.join()
    
        
if __name__ == '__main__':
    main()