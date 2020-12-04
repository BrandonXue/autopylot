# Local modules
from typing import List
from config import *
from defs import *
from rectangle import *

# Other modules
import cv2
import pygame
import pygame.freetype
import numpy as np

# Specific components
from math import sqrt, radians, sin, cos, fabs
from random import randint
    
def clamp(value, min, max):
    if value > max:
        value = max
    elif value < min:
        value = min
    return value

class CarGame:
    def __init__(self, game_mode: str='play', flip: bool=True):
        self.game_mode = game_mode
        self.flip = flip

        # We may want to init() selective modules (to look into if we have time)
        pygame.init()

        # Game environment related
        self.world_bounds = Rectangle(-MAX_X_LOC_BOX, -MAX_Y_LOC_BOX, MAP_WIDTH, MAP_HEIGHT, RGB_WHITE)
        self.points = 0
        self.reward = 0 # for reinforcement learning
        self.frame_count = 0

        # Keyboard related
        self.key_list = [False] * KEY_ARRAY_SIZE # Lists are faster for simple random access

        self.clock = pygame.time.Clock() # For FPS

        # Display related
        self.display = pygame.display.set_mode(
            VIEWPORT_SIZE,
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
        )
        self.display.set_alpha(None)

        self.cover_display = pygame.Surface((GRAYSCALE_DIM, GRAYSCALE_DIM))

        self.font = pygame.freetype.Font(None, 14)
    
        # Pre-calculated game constants 
        self.half_viewport = pygame.math.Vector2(VIEWPORT_SIZE)/2
        self.display_offset = self.half_viewport
        self.display_diag = sqrt(self.half_viewport[0]**2 + self.half_viewport[1]**2)

        self.collision_check_dist = (
            sqrt((CAR_WIDTH/2)**2 + (CAR_HEIGHT/2)**2) # player/car box hypoetenuse
            + sqrt((MAX_BOX_WIDTH/2)**2 + (MAX_BOX_HEIGHT/2)**2) # terrain object max hypotenuse
            + 20 # Arbitrary number to ensure we don't filter out anything that looks like it should be colliding
        )

        # Frame counter modulus for throttling certain events
        self.num_frames_per_batch = 4
        
    def set_learner(self, learner):
        self.learner = learner

    def reset_map(self):
        self.mark_reset = False
        self.running_reward = 0 # running reward of model
        self.episode_count = 0
        self.learning_fc = 0 # model frame count
        self.points = 0
        self.reward = 0
        self.rotation = 0.0
        self.angular_velocity = 0.0 # Can think of this as steering wheel position
        self.velocity = 0.0

        self.player_rect = PlayerRectangle(
            self.player_start_pos[0]*MAX_BOX_WIDTH + MAX_BOX_WIDTH/2, 
            self.player_start_pos[1]*MAX_BOX_HEIGHT + MAX_BOX_HEIGHT/2, 
            CAR_WIDTH, CAR_HEIGHT
        )
        self.add_goal_rects_from_pool() # Generate new goal rectangles

        self.draw_game_scene() # Draw the game scene after reset

        # Fill the buffer with four of the initial screen to make sure all non null
        if self.game_mode == 'train':
            self.state_buffer = [] 
            for iteration in range(4):
                self.state_buffer.append(self.get_processed_frame())
        
    def load_map_from_file(self, filepath):
        in_file = open(filepath) # default read only
        
        self.rects = []
        self.reward_dot_pool = []

        i = 0
        for line in in_file.readlines():
            j = 0
            for object_code in line:
                if object_code == '0':
                    self.reward_dot_pool.append((i, j))
                elif object_code == '1':
                    self.rects.append(
                        EnvironmentRectangle(
                            i*MAX_BOX_WIDTH, j*MAX_BOX_HEIGHT, # Location coords
                            MAX_BOX_WIDTH, MAX_BOX_HEIGHT, RGB_DARK_GRAY # width height
                        )
                    )
                elif object_code == '2':
                    self.player_start_pos = (i, j)
                    self.player_rect = PlayerRectangle(
                        i*MAX_BOX_WIDTH + MAX_BOX_WIDTH/2, 
                        j*MAX_BOX_HEIGHT + MAX_BOX_HEIGHT/2, 
                        CAR_WIDTH, CAR_HEIGHT
                    )   
                j = j+1
            i = i+1
        in_file.close()

    def add_goal_rects_from_pool(self): #needs set alive function
        self.goal_rects = []
        temp_pool = self.reward_dot_pool.copy()
        for i in range(0, NUMBER_OF_DOTS):
            location = temp_pool[randint(0, len(temp_pool)-1)]
            temp_pool.remove(location)
            self.goal_rects.append(
                GoalRectangle(
                    (location[0] * MAX_BOX_WIDTH) + (MAX_BOX_WIDTH - 60) / 2,
                    (location[1] * MAX_BOX_HEIGHT) + (MAX_BOX_HEIGHT - 60) / 2, 
                    60, 60, RGB_BLACK, RGB_WHITE
                )
            )

    def store_input_keys(self):
        ''' Stores keyboard events in self.key_list and also stores QUIT as ESCAPE. '''

        # In play mode listen for all input keys within KEY_ARRAY_SIZE
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Treat quit event as escape button pressed
                self.key_list[pygame.K_ESCAPE] = True
            
            # Allother keyboard inputs
            elif event.type == pygame.KEYDOWN and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = True
            elif event.type == pygame.KEYUP and event.key < KEY_ARRAY_SIZE:
                self.key_list[event.key] = False

    def store_exit_events(self):
        ''' Only listens for events that quit the game and stores them as keys. '''
        
        for event in pygame.event.get():
            if ((event.type == pygame.QUIT) 
                or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                    self.key_list[pygame.K_ESCAPE] = True
                    return

    def update_positions(self):
        pressed = False

        if self.key_list[pygame.K_w]:
            self.velocity -= CAR_ACCELERATION
            pressed=True
            
        if self.key_list[pygame.K_s]:
            self.velocity += CAR_ACCELERATION 
            pressed=True
            
        # Turning left only
        if self.key_list[pygame.K_a] and not self.key_list[pygame.K_d]:
            self.angular_velocity = clamp(
                self.angular_velocity + CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Turning right only 
        elif self.key_list[pygame.K_d] and not self.key_list[pygame.K_a]:
            self.angular_velocity = clamp(
                self.angular_velocity - CAR_ANGULAR_ACCELERATION, # new value
                -CAR_MAX_ANGULAR_VELOCITY, CAR_MAX_ANGULAR_VELOCITY # clamp bounds
            )

        # Else either both pressed or neither pressed. Move wheel towards center.
        else:
            if (self.angular_velocity < CAR_ANGULAR_ZERO_THRESHOLD 
                and self.angular_velocity > -CAR_ANGULAR_ZERO_THRESHOLD):
                self.angular_velocity = 0.0
            if self.angular_velocity != 0.0:
                self.angular_velocity *= CAR_ANGULAR_VELOCITY_DECAY
            
        # Only turn if car is in motion
        if self.velocity != 0:
            # Make turn speed dependent on velocity
            self.rotation -= self.angular_velocity * self.velocity

        self.velocity = clamp(self.velocity, -CAR_MAX_VELOCITY, CAR_MAX_VELOCITY)
        
        if not pressed:
            if self.velocity < CAR_VELOCITY_ZERO_THRESHOLD and self.velocity > -CAR_VELOCITY_ZERO_THRESHOLD:
                self.velocity = 0.0
            if self.velocity != 0:
                self.velocity *= CAR_VELOCITY_DECAY

        # Move player position, (you can add tuple to Vector2)
        rads = radians(self.rotation)
        self.player_rect.move( (sin(rads) * self.velocity, cos(rads) * self.velocity) )


    def draw_game_scene(self):
        # These locations are used as references
        player_center = self.player_rect.get_center()
        rear_axle = self.player_rect.get_rear_center()
        
        #Draw world boundaries
        self.world_bounds.pivot_and_offset(rear_axle, self.rotation, self.display_offset)

        # Render the player
        player_coords = self.player_rect.pivot_and_offset(rear_axle, 0, self.display_offset)
        self.player_rect.draw(self.display, player_coords)

        self.collision = False # handle_rect_list may set to true
        self.handle_rect_list(self.rects, player_center, player_coords, rear_axle)
        self.reached_goal = False # handle_rect_list may set to true
        self.handle_goal_list(self.goal_rects, player_center, player_coords, rear_axle)

    def handle_goal_list(self, goal_list, player_center, player_coords, rear_axle):
        for obj in goal_list:
                
            if obj.get_is_alive():
                obj_center = obj.get_center()
                max_viewable_dist = self.display_diag + obj.get_max_dim()

                dist_between = player_center.distance_to(obj_center)
                if (dist_between < max_viewable_dist):

                    object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                    donut_coords = obj.pivot_and_offset_donut(rear_axle, self.rotation, self.display_offset)

                    if (dist_between < self.collision_check_dist 
                        and check_collision(player_coords, object_coords)):
                            self.points += 1
                            self.reached_goal = True # Used for reward assignment
                            obj.set_is_alive(False)
                                #self.update_goal_rect()
                    
                    obj.draw(self.display, object_coords)
                    obj.draw_donut(self.display, donut_coords)

    def handle_rect_list(self, rect_list, player_center, player_coords, rear_axle):
        for obj in rect_list:
                
            if obj.get_is_alive():
                obj_center = obj.get_center()
                max_viewable_dist = self.display_diag + obj.get_max_dim()

                dist_between = player_center.distance_to(obj_center)
                if (dist_between < max_viewable_dist):

                    object_coords = obj.pivot_and_offset(rear_axle, self.rotation, self.display_offset)
                    
                    if (not self.collision and dist_between < self.collision_check_dist):
                        if check_collision(player_coords, object_coords):
                            self.collision = True
                            self.mark_reset = True # Used for reward assignment
                    
                    obj.draw(self.display, object_coords)

    def interpret_actions(self, actions):
        ''' Interprets an action list as [W, A, S, D] '''
        if actions == 0: # forward
            self.key_list[pygame.K_w] = True
            self.key_list[pygame.K_a] = False
            self.key_list[pygame.K_s] = False
            self.key_list[pygame.K_d] = False

        elif actions == 1: # Left forward
            self.key_list[pygame.K_w] = True
            self.key_list[pygame.K_a] = True
            self.key_list[pygame.K_s] = False
            self.key_list[pygame.K_d] = False

        elif actions == 3: # Right forward
            self.key_list[pygame.K_w] = True
            self.key_list[pygame.K_a] = False
            self.key_list[pygame.K_s] = False
            self.key_list[pygame.K_d] = True

        elif actions == 2: # backward
            self.key_list[pygame.K_w] = False
            self.key_list[pygame.K_a] = False
            self.key_list[pygame.K_s] = True
            self.key_list[pygame.K_d] = False

        elif actions == 4: # left backward
            self.key_list[pygame.K_w] = False
            self.key_list[pygame.K_a] = True
            self.key_list[pygame.K_s] = True
            self.key_list[pygame.K_d] = False

        elif actions == 5: # right backward
            self.key_list[pygame.K_w] = False
            self.key_list[pygame.K_a] = False
            self.key_list[pygame.K_s] = True
            self.key_list[pygame.K_d] = True

        elif actions == 4: # left
            self.key_list[pygame.K_w] = False
            self.key_list[pygame.K_a] = True
            self.key_list[pygame.K_s] = False
            self.key_list[pygame.K_d] = False

        elif actions == 5: # right
            self.key_list[pygame.K_w] = False
            self.key_list[pygame.K_a] = False
            self.key_list[pygame.K_s] = False
            self.key_list[pygame.K_d] = True

    def draw_observation(self, pixels):
        pygame.surfarray.blit_array(self.cover_display, pixels)

    def draw_play_dashboard(self):
        points_text = f"Points: {self.points:.2f}"
        fps_text = f"FPS: {self.fps:.2f}"
        self.font.render_to(self.display, (4, 4), points_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (4, VIEWPORT_HEIGHT-32), fps_text, RGBA_LIGHT_RED)

    def draw_train_dashboard(self):
        if not self.cover_display.get_locked() and not self.display.get_locked():
            cover_display_area = self.display.blit(
                self.cover_display, (VIEWPORT_WIDTH-GRAYSCALE_DIM-8, VIEWPORT_HEIGHT-GRAYSCALE_DIM-8)
            )
        else:
            cover_display_area = None
        
        reward_text = f"Reward: {self.reward:.2f},    Running Reward: {self.running_reward:.2f}"
        training_text = f"Episode Count: {self.episode_count:.2f},    Frame Count: {self.learning_fc:.2f}"
        fps_text = f"FPS: {self.fps:.2f}"
        self.font.render_to(self.display, (4, 4), reward_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (4, 32), training_text, RGBA_LIGHT_RED)
        self.font.render_to(self.display, (4, VIEWPORT_HEIGHT-32), fps_text, RGBA_LIGHT_RED)

        return cover_display_area

    def set_dashboard_info(self, info):
        self.running_reward = info[0] # running reward
        self.episode_count = info[1] # episode count
        self.learning_fc = info[2] # frame count

    def calc_reward(self):
        ''' Ideas: 
        Punish indecisiveness (flip-flopping inputs)
        Punish idleness (low velocity for extended durations of time)
        Heavily punish crashing (when collision == True)
        Reward exploration (reward cumulative velocity magnitude)
        '''

        if self.mark_reset:
            self.reward = -1
        elif self.reached_goal:
            self.reward = 1
        else:
            self.reward = 0.3 * fabs(self.velocity) / CAR_MAX_VELOCITY

    def playing_game_loop(self):
        while True:
            # Mange framerate
            self.clock.tick(FRAME_RATE)
            self.fps = self.clock.get_fps()

            self.store_input_keys() # Get input events
            
            if self.key_list[pygame.K_ESCAPE]: # If escape is pressed, break main event loop
                exit(0)

            # Fill background color
            self.display.fill(RGB_WHITE)

            self.draw_game_scene()
            self.update_positions() # Update movable object positions
            self.draw_play_dashboard()

            pygame.display.flip()

            if self.mark_reset == True:
                self.reset_map()

    def get_state(self):
        '''
        Get a state from the game environment
        Returns only observations.
        '''

        return self.state_buffer

    def get_event_step(self, actions, info):
        '''
        Advance the game environment by one step, applying the actions given.
        Any info given will be drawn to the screen
        Returns (observations, reward, done, info).
        '''
        
        # Mange framerate
        self.clock.tick(FRAME_RATE)
        self.fps = self.clock.get_fps()

        # Get input events
        self.store_exit_events()

        # If escape is pressed, break main event loop
        if self.key_list[pygame.K_ESCAPE]:
            self.display.fill(RGB_WHITE) # Reset screen to white
            save_text = f"SAVING... DO NOT FORCE EXIT"
            self.font.render_to(self.display, (VIEWPORT_WIDTH/2, VIEWPORT_HEIGHT/2), save_text, RGB_RED)
            pygame.display.flip()
            self.learner.save_items()
            exit(0)

        self.display.fill(RGB_WHITE) # Reset screen to white

        # Handle game environment display
        self.interpret_actions(actions)

        self.update_positions() # Update movable object positions

        self.draw_game_scene()  # Draw player,other rectangles, etc
                                # This will update self.reached_goal

        # Push new frame to buffer before updating dashboard
        self.push_frame_to_buffer()

        # Now update dashboard
        self.set_dashboard_info(info)
        self.draw_observation(self.state_buffer[2])
        dashboard_area = self.draw_train_dashboard()

        self.calc_reward() # Update reward based on previous actions

        if self.flip:
            pygame.display.flip()
        elif dashboard_area:
            pygame.display.update(dashboard_area)

        return (self.get_state(), self.reward, self.mark_reset, None)

    def push_frame_to_buffer(self):
        ''' Add a new frame to the end of the frame_buffer.
        Does not change overall length of buffer.'''

        # Append a new frame
        self.state_buffer.append(self.get_processed_frame())
        del self.state_buffer[:1]

    def get_processed_frame(self):
        ''' Return a snapshot of the current display after processing. '''

        pixels = np.array(pygame.surfarray.array3d(self.display), dtype='float32')
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        pixels = cv2.resize(pixels, (GRAYSCALE_DIM, GRAYSCALE_DIM))
        return pixels