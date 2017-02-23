# Coding group: Alexandre,... 
import gym
import gym_pull #to import the environment of doom

gym_pull.pull('github.com/ppaquette/gym-doom')
env = gym.make('ppaquette/DoomBasic-v0') 