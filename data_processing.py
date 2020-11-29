
# import tensorflow as tf

# from tensorflow import keras

class DataProcessor:
    def __init__(self, pipe_conn, exit_keyword):
        self.pipe_conn = pipe_conn
        self.exit_keyword = exit_keyword

    def close_pipe(self):
        self.pip_conn.close()

    def start(self):
        while True:
            if self.pipe_conn.poll():
                received = self.pipe_conn.recv()
                if received == self.exit_keyword:
                    break

                # TODO: Train neural net here
                print(received)
