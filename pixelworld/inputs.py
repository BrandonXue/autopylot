# Authors:
# Brandon Xue       brandonx@csu.fullerton.edu
# Jacob Rapmund     jacobwrap86@csu.fullerton.edu
#
# This module contains encapsulations for handling keyboard
# inputs and event. It uses pygame's functionality.

# Non-local modules
import pygame.event
from pygame.constants import * 

# pygame constants will be accessible externally, e.g. inputs.K_w

class GameKeys1:
    ''' 
    Game keys, one at a time.
    An encapsulation of pygame input functionality intended for games.
    If more than one WASD or Arrow Key is pressed at roughly the same time,
    the latest pressed takes precedence.
    '''

    def __init__(self):
        self.__quit = False
        self.__move_key_buffer = []

    def grab_keys(self):
        ''' 
        Grabs input events and stores keyboard state.
        Arrow keys are stored as WASD
        '''

        for event in pygame.event.get():
            # Keydown events
            if event.type == KEYDOWN:
                if event.key == K_a or event.key == K_LEFT:
                    self.__move_key_buffer += [K_a]
                elif event.key == K_w or event.key == K_UP:
                    self.__move_key_buffer += [K_w]
                elif event.key == K_d or event.key == K_RIGHT:
                    self.__move_key_buffer += [K_d]
                elif event.key == K_s or event.key == K_DOWN:
                    self.__move_key_buffer += [K_s]
                elif event.key == K_ESCAPE:
                    self.__quit = True
            # Keyup events
            elif event.type == KEYUP:
                try:
                    if event.key == K_a or event.key == K_LEFT:
                        self.__move_key_buffer.remove(K_a)
                    elif event.key == K_w or event.key == K_UP:
                        self.__move_key_buffer.remove(K_w)
                    elif event.key == K_d or event.key == K_RIGHT:
                        self.__move_key_buffer.remove(K_d)
                    elif event.key == K_s or event.key == K_DOWN:
                        self.__move_key_buffer.remove(K_s)
                except: # Key wasn't in buffer... somehow
                    pass
            # Quit event
            elif event.type == QUIT:
                self.__quit = True

    def get_move_key(self):
        ''' 
        Returns the last movement key that was pressed, or None.
        For comparison, use WASD instead of arrow keys.
        '''

        if len(self.__move_key_buffer) > 0:
            return self.__move_key_buffer[-1]
        else:
            return None

    def has_quit(self):
        ''' Returns True if the escape key was pressed or there was a quit event. '''
        return self.__quit

class QuitKeys:
    ''' 
    Game keys, but only listening for escape or quit events.
    An encapsulation of pygame input functionality intended for automated games.
    If you only need escape or quit events, this is more lightweight.
    '''

    def __init__(self):
        self.__quit = False

    def grab_keys(self):
        ''' Grabs input events, only looking for escape or quit. '''

        for event in pygame.event.get():
            if event.type == QUIT:
                self.__quit = True
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.__quit = True
    
    def has_quit(self):
        ''' Returns True if the escape key was pressed or there was a quit event. '''
        return self.__quit
