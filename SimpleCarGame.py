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
    KEY_ARRAY_SIZE = 1024
    keys = np.array([False]*KEY_ARRAY_SIZE)
    viewport_size = width, height = 1080, 720
    viewport = pygame.display.set_mode(viewport_size)
    screen_rect = viewport.get_rect()
    
    map_size = width, height = 2000, 2000
    
    #some colors
    black = 0, 0, 0
    blue = 0, 0, 255
    green = 0, 255, 0
    red = 255, 0, 0
    
    rotation = 0.0
    angular_velocity = 0.0 # Can think of this as steering wheel position
    velocity = 0.0
    acceleration = 0.005
    pressed = False

    MAX_VELOCITY_MAGNITUDE = 2.0
    
    #object position    x, y
    abs_pos = np.array([0, 0])
    
    player_test = rectangle(0, 0, 20, 40)
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
            velocity -= acceleration
            pressed=True
            
        if keys[pygame.K_s]:
            velocity += acceleration 
            pressed=True
            
        # Turning left only
        if keys[pygame.K_a] and not keys[pygame.K_d]:
            angular_velocity = clamp(angular_velocity+0.002, -0.4, 0.4)

        # Turning right only 
        elif keys[pygame.K_d] and not keys[pygame.K_a]:
            angular_velocity = clamp(angular_velocity-0.002, -0.4, 0.4)

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if angular_velocity < 0.001 and angular_velocity > -0.001: angular_velocity = 0.0
            if angular_velocity != 0.0: angular_velocity *= 0.985
            
        # Only turn if car is in motion
        if velocity != 0:
            # Make turn speed dependent on velocity
            rotation -= angular_velocity * velocity

        velocity = clamp(velocity, -MAX_VELOCITY_MAGNITUDE, MAX_VELOCITY_MAGNITUDE)
        move[0] += sin(rads) * velocity
        move[1] += cos(rads) * velocity
        
        if not pressed:
            if velocity < 0.01 and velocity > -0.01: velocity = 0.0
            if velocity != 0: velocity *= 0.99
                
        pressed = False
        
        player_test.move(move)
        view_pos += move
        
        #screen reset
        viewport.fill(black)
        
        #render
        radian_val = radians(rotation)
        rotation_matrix = np.array([[cos(radian_val), -sin(radian_val)], [sin(radian_val), cos(radian_val)]])

        player_view_adjust = np.array(viewport_size)/2 + player_test.get_front_center() - view_pos
        view_ajdust = np.array(viewport_size)/2 + player_test.get_center() - view_pos

        pygame.draw.polygon(viewport, blue, player_test.get_coords2(player_test.get_front_center(), player_rotation_matrix, player_view_adjust))
        for object in objects:
            pygame.draw.polygon(viewport, green, object.get_coords2(player_test.get_center(), rotation_matrix, view_ajdust))
        
        #swap buffers
        pygame.display.flip()
        
if __name__ == '__main__':
    main()