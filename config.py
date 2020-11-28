# ============== USER AND REMINDER ===============
CURRENT_USER = 'Brandon Xue'
# CURRENT_USER = 'Jacob Rapmund'
print('\nConfig loaded for', CURRENT_USER, end='\n\n') # Remind on load

# =================== VIEWPORT ===================
VIEWPORT_WIDTH = {
    'Brandon Xue': 1080,
    'Jacob Rapmund': 1080
}[CURRENT_USER]

VIEWPORT_HEIGHT = {
    'Brandon Xue': 720,
    'Jacob Rapmund': 720
}[CURRENT_USER]

VIEWPORT_SIZE = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)

# ====================== MAP =====================
MAP_WIDTH = {
    'Brandon Xue': 2000,
    'Jacob Rapmund': 2000
}[CURRENT_USER]

MAP_HEIGHT = {
    'Brandon Xue': 2000,
    'Jacob Rapmund': 2000
}[CURRENT_USER]

MAP_SIZE = (MAP_WIDTH, MAP_HEIGHT)

# ====================== CAR =====================
CAR_WIDTH = {
    'Brandon Xue': 20,
    'Jacob Rapmund': 20
}[CURRENT_USER]

CAR_HEIGHT = {
    'Brandon Xue': 40,
    'Jacob Rapmund': 40
}[CURRENT_USER]

CAR_SIZE = (CAR_WIDTH, CAR_HEIGHT)

CAR_ACCELERATION = {
    'Brandon Xue': 0.005,
    'Jacob Rapmund': 0.002
}[CURRENT_USER]

CAR_MAX_VELOCITY = {
    'Brandon Xue': 2.0,
    'Jacob Rapmund': 0.8
}[CURRENT_USER]

CAR_VELOCITY_ZERO_THRESHOLD = {
    'Brandon Xue': 0.01,
    'Jacob Rapmund': 0.01
}[CURRENT_USER]

CAR_VELOCITY_DECAY = {
    'Brandon Xue': 0.99,
    'Jacob Rapmund': 0.993
}[CURRENT_USER]

CAR_ANGULAR_ACCELERATION = {
    'Brandon Xue': 0.002,
    'Jacob Rapmund': 0.001
}[CURRENT_USER]

CAR_MAX_ANGULAR_VELOCITY = {
    'Brandon Xue': 0.4,
    'Jacob Rapmund': 0.4
}[CURRENT_USER]

CAR_ANGULAR_ZERO_THRESHOLD = {
    'Brandon Xue': 0.001,
    'Jacob Rapmund': 0.001
}[CURRENT_USER]

CAR_ANGULAR_VELOCITY_DECAY = {
    'Brandon Xue': 0.985,
    'Jacob Rapmund': 0.990
}[CURRENT_USER]

# ===================== OTHER ====================
KEY_ARRAY_SIZE = {
    'Brandon Xue': 1024,
    'Jacob Rapmund': 1024
}[CURRENT_USER]