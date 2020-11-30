from config import *
from defs import ExitSignalType

import cv2
import dill
import mss
import numpy as np
# import tensorflow as tf

# from tensorflow import keras

class DataProcessor:
    def __init__(self, read_conn, write_conn):
        self.read_conn = read_conn
        self.write_conn = write_conn

    def start(self):
    
        #with mss.mss() as sct:
        #    monitor = sct.monitors[1]
        #    pixels = np.array(sct.grab(monitor))
        #    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
        #    pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))
        #    pixels = pixels.astype('float64')
            
        sct = mss.mss()
        mon = sct.monitors[1]
        
        monitor = {
            "top": mon["top"],  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": VIEWPORT_WIDTH,
            "height": VIEWPORT_HEIGHT,
            "mon": 1,
        }
            
        while True:
            if self.read_conn.poll():
                received = self.read_conn.recv()

                # If exit signal received, perform handshake and return
                if type(received) is ExitSignalType:
                    self.write_conn.send(ExitSignalType())
                    return True
                
            # TODO: Train neural net here
            pixels = np.array(sct.grab(monitor))
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
            pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))
            #pixels = np.transpose(pixels.astype('float64'), (1,0,2))
            pixels = np.transpose(pixels.astype('float64'), (1,0))
            
            if dill.pickles(pixels):
                self.write_conn.send(pixels)