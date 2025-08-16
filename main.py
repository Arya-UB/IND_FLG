from turtledemo.nim import SCREENWIDTH, SCREENHEIGHT

import pygame
import sys
import random
import  threading
import pyttsx3
import time

# initializing pygame and pyttsx3

pygame.init()
engine = pyttsx3.init()

# screen setup
SCREEN_WIDTH,SCREEN_HEIGHT = 800,600
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("Catch The Flag With Pappu")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
ORANGE = (255, 165, 0)
SAPPHIRE = (15, 82, 186)

# Fonts
font_medium = pygame.font.SysFont('Arial', 28)
font_small = pygame.font.SysFont('Arial', 22)

# Constants
PLAYER_SIZE = 50
FLAG_SIZE = 40
TIME_LIMIT = 60
PLAYER_SPEED = 5




