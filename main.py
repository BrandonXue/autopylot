from car_game import CarGame
from data_processing import DataProcessor

from multiprocessing import Process, Pipe

def start_data_processing(pipe_connection, exit_keyword):
    data_pro = DataProcessor(pipe_connection, exit_keyword)
    data_pro.start()

def start_game(pipe_connection, exit_keyword):
    car_game = CarGame(pipe_connection, exit_keyword)
    car_game.start()

def main():
    exit_keyword = 'exit'   # define a message to be sent from game_proc to data_proc to terminate
    #read_end, write_end = Pipe(False)   # use a pipe for inter-process communication
    read_end, write_end = Pipe(True)

    # Create a separate process for the game and for data processing
    data_proc = Process(target=start_data_processing, args=(read_end, exit_keyword))
    game_proc = Process(target=start_game, args=(write_end, exit_keyword))

    # Start both processes
    data_proc.start()
    game_proc.start()
    

    # Wait for processes to finish
    data_proc.join()
    game_proc.join()
    
        
if __name__ == '__main__':
    main()