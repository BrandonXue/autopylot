from defs import *
import pygame, pygame.freetype

#click states
NONE = 0
PRESS = 1
RELEASE = 2
CLICK = 3

class Button:
    def __init__(self, dimensions, context, pos = [0,0], forecolor=RGB_GRAY, backcolor=RGB_GRAY, text="", fontsize=12, fontcolor=RGB_BLACK, borderwidth=0):
        self.context_ = context
        self.image_ = pygame.Surface(list(dimensions), pygame.SRCALPHA)
        self.dimensions_ = dimensions
        self.forecolor_ = forecolor
        self.backcolor_ = backcolor
        self.text_ = text
        self.fontsize_ = fontsize
        self.fontcolor_ = fontcolor
        self.font_ = pygame.freetype.Font(None, self.fontsize_)
        self.borderwidth_ = borderwidth
        self.pos_ = pos
        self.on_click_action_ = None
        self.on_press_action_ = None
        self.on_release_action_ = None
        
        self.context_.add_listener(self)
        
    def set_text(self, text):
        self.text_ = text
    
    def set_font_size(self, size):
        self.fontsize_ = size
        self.font_ = pygame.freetype.Font(None, self.fontsize_)
        
    def set_backcolor(self, nColor):
        self.backcolor_ = nColor
        
    def remove_from_observable(self):
        self.context_.remove_listener(self)
        self.context_ = None
        
    def add_to_observable(self, context):
        self.context_ = context
        self.context_.add_listener(self)
    
    def update(self, state, mouse_pos):
        if mouse_pos[0] >= self.pos_[0] and mouse_pos[1] >= self.pos_[1] and mouse_pos[0] <= self.pos_[0] + self.dimensions_[0] and mouse_pos[1] <= self.pos_[1] + self.dimensions_[1]:
            if state == PRESS and self.on_press_action_ != None:
                self.on_press_action_()
            elif state == RELEASE and self.on_release_action_ != None:
                self.on_release_action_()
            elif state == CLICK and self.on_click_action_ != None:
                self.on_click_action_()
    
    #pass in lambda
    def on_click(self, action):
        self.on_click_action_ = action
        
    #pass in lambda
    def on_press(self, action):
        self.on_press_action_ = action
        
    #pass in lambda
    def on_release(self, action):
        self.on_release_action_ = action
        
    def draw(self):
        if self.context_ != None:
            self.image_.fill(self.backcolor_)
            
            #Centers text
            dims = self.font_.get_rect(self.text_).size
            pos = (self.dimensions_[0]//2 - dims[0]//2, self.dimensions_[1]//2 - dims[1]//2)
            ############
            
            self.font_.render_to(self.image_, pos, self.text_, self.fontcolor_)
            self.context_.get_image().blit(self.image_, self.pos_)
            
class TextField:
    def __init__(self, pos, text = "", fontsize = 32, fontcolor = RGB_BLACK):
        self.pos_ = pos
        self.text_ = text
        self.fontsize_ = fontsize
        self.fontcolor_ = fontcolor
        self.context_ = None
        
        self.font_ = pygame.freetype.Font(None, self.fontsize_)
        
    def set_text(self, text):
        self.text_ = text
    
    def set_font_size(self, size):
        self.fontsize_ = size
        self.font_ = pygame.freetype.Font(None, self.fontsize_)
        
    def set_backcolor(self, nColor):
        self.backcolor_ = nColor
        
    def set_context(self, context):
        self.context_ = context
        
    def draw(self):
        if self.context_ != None:
            self.font_.render_to(self.context_.get_image(), self.pos_, self.text_, self.fontcolor_)
            
class Menu:
    def __init__(self, dimensions, pos, color):
        self.image_ = pygame.Surface(list(dimensions), pygame.SRCALPHA)
        self.listeners_ = []
        self.textfields_ = {}
        self.color_ = color
        self.state_ = NONE
        self.mouse_pos_ = [0,0]
        self.pos_ = pos
        
    def add_listener(self, button):
        self.listeners_.append(button)
        
    def remove_listener(self, button):
        self.listeners_.remove(button)
        
    def add_text_field(self, id, text_field):
        text_field.set_context(self)
        self.textfields_[id] = text_field
        
    def get_text_field(self, id):
        return self.textfields_[id]
        
    def set_text_field(self, id, text):
        self.textfields_[id].set_text(text)
        
    def get_image(self):
        return self.image_
        
    def notify(self):
        for button in self.listeners_:
            button.update(self.state_, self.mouse_pos_)
            
    def set_state(self, state):
        self.state_ = state
        self.notify()
        
    def set_mouse_pos(self, mouse_pos):
        self.mouse_pos_[0] = mouse_pos[0] - self.pos_[0]
        self.mouse_pos_[1] = mouse_pos[1] - self.pos_[1]
        
    def draw(self, viewport):
        self.image_.fill(self.color_)
        for button in self.listeners_:
            button.draw()
            
        for text_field in self.textfields_.values():
            text_field.draw()
        viewport.blit(self.image_, self.pos_)
        
        
