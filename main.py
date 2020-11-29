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
    conn1, conn2 = Pipe()   # use a pipe for inter-process communication

    # Create a separate process for the game and for data processing
    game_proc = Process(target=start_game, args=(conn1, exit_keyword))
    data_proc = Process(target=start_data_processing, args=(conn2, exit_keyword))

    # Start both processes
    game_proc.start()
    data_proc.start()

    # Wait for processes to finish
    game_proc.join()
    data_proc.join()
        
if __name__ == '__main__':
    main()