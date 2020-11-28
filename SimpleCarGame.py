from config import *
from rgb_colors import *
import sys, pygame
import tensorflow as tf
from tensorflow import keras
import numpy as np
from math import sqrt, radians, sin, cos, exp, fabs
from random import randint
    
def rotate_around_player2(player_pos, obj_pos, rotation_matrix: np.array):
    temp_vec = obj_pos - player_pos
    return rotation_matrix.dot(temp_vec)

def rotate_around_player(player_pos, obj_pos, rotation)->np.array:
    temp_vec = obj_pos - player_pos
    radian_val = radians(rotation)
    rotation_matrix = np.array([[cos(radian_val), -sin(radian_val)], [sin(radian_val), cos(radian_val)]])
    rotated_position = rotation_matrix.dot(temp_vec) + player_pos
    return rotated_position
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class rectangle:
    def __init__(self, x=0, y=0, width=10, height=10):
        self.coordinates_ = np.array([[x, y],[x+width, y], [x+width, y+height], [x, y+height]], dtype=float)
        self.center_ = np.array([x+width/2,y+height/2], dtype=float)
        self.front_center_ = np.array([x+width/2, y], dtype=float)

    def get_coords2(self, point_of_rotation, rotation_matrix, view_adjust):
        new_coords = []
        for point in self.coordinates_:
            new_coords.append(rotation_matrix.dot(point - point_of_rotation) + view_adjust)
        return new_coords
        
    def move(self, move_values:np.array(2) = np.array([0.,0.])):
        self.coordinates_ += move_values
        self.center_ += move_values
        self.front_center_ += move_values
        
    def get_center(self):
        return self.center_
        
    def get_front_center(self):
        return self.front_center_

def main():
    pygame.init()
    keys = np.array([False]*KEY_ARRAY_SIZE)
    viewport = pygame.display.set_mode(VIEWPORT_SIZE)
    half_viewport = np.array(VIEWPORT_SIZE)/2
    screen_rect = viewport.get_rect()
    
    rotation = 0.0
    angular_velocity = 0.0 # Can think of this as steering wheel position
    velocity = 0.0
    pressed = False
    
    #object position    x, y
    abs_pos = np.array([0, 0])
    
    player_test = rectangle(0, 0, CAR_WIDTH, CAR_HEIGHT)
    #object_test = rectangle(-100, -100, 50, 50)
    objects = list()
    
    for i in range(0, 10):
        objects.append(rectangle(randint(-750,750), randint(-750, 750), randint(25, 75), randint(25, 75)))
    
    #viewport position needs to center on object
    view_pos = np.array([0., 0.])
    player_rotation_matrix = np.array([[1.0, 0.0], [0.0, 1.0]]) # I2

    while True:
        #update events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
                keys[event.key] = True
            elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
                keys[event.key] = False
            
        #game logic
        move = np.array([0., 0.])
        rads = radians(rotation)
        
        if keys[pygame.K_w]:
            velocity -= CAR_ACCELERATION
            pressed=True
            
        if keys[pygame.K_s]:
            velocity += CAR_ACCELERATION 
            pressed=True
            
        # Turning left only
        if keys[pygame.K_a] and not keys[pygame.K_d]:
            angular_velocity = clamp(
                angular_velocity+CAR_ANGULAR_ACCELERATION,
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
            )

        # Turning right only 
        elif keys[pygame.K_d] and not keys[pygame.K_a]:
            angular_velocity = clamp(
                angular_velocity-CAR_ANGULAR_ACCELERATION,
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY
            )

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD and angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD:
                angular_velocity = 0.0
            if angular_velocity != 0.0:
                angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
        # Only turn if car is in motion
        if velocity != 0:
            # Make turn speed dependent on velocity
            rotation -= angular_velocity * velocity

        velocity = clamp(velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
        move[0] += sin(rads) * velocity
        move[1] += cos(rads) * velocity
        
        if not pressed:
            if velocity < CAR_VELOCITY_ZERO_THRESHOLD and velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                velocity = 0.0
            if velocity != 0:
                velocity *= CAR_VELOCITY_DECAY
                
        pressed = False
        
        player_test.move(move)
        view_pos += move
        
        #screen reset
        viewport.fill(RGB_BLACK)
        
        #render
        rotation_matrix = np.array([[cos(rads), -sin(rads)], [sin(rads), cos(rads)]])

        player_view_adjust = half_viewport + player_test.get_front_center() - view_pos
        view_ajdust = half_viewport + player_test.get_center() - view_pos

        pygame.draw.polygon(viewport, RBG_BLUE, player_test.get_coords2(player_test.get_front_center(), player_rotation_matrix, player_view_adjust))
        for object in objects:
            pygame.draw.polygon(viewport, RGB_GREEN, object.get_coords2(player_test.get_center(), rotation_matrix, view_ajdust))
        
        #swap buffers
        pygame.display.flip()
        
if __name__ == '__main__':
    main()