from config import *
from skimage import color, transform, exposure
import dill
# import tensorflow as tf

# from tensorflow import keras

class DataProcessor:
    def __init__(self, pipe_conn, exit_keyword):
        self.pipe_conn = pipe_conn
        self.exit_keyword = exit_keyword

    def close_pipe(self):
        self.pipe_conn.close()

    def start(self):
    
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            print(sct_img)
            
            
        while True:
            if self.pipe_conn.poll():
                received = self.pipe_conn.recv()
                if received == self.exit_keyword:
                    break

                # TODO: Train neural net here
                print(received)
                pixels = color.rgb2gray(received)
                pixels = transform.resize(pixels,(GRAYSCALE_DIM,GRAYSCALE_DIM))
                pixels = exposure.rescale_intensity(pixels, out_range=(0, 255))
                print(pixels)
                
                if dill.pickles(pixels):
                    self.pipe_conn.send(pixels)
                else:
                    print('pixels not sent. Not pickleable.')
                    
                print("pixels sent")
