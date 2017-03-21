# Coding group: Alexandre,... 
import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from matplotlib import pyplot as plt
import tensorflow
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.models import Sequential

	
#https://elitedatascience.com/keras-tutorial-deep-learning-in-python

# =============================
#   ATARI Environment Wrapper
#   I took the class from this link
#   https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py
# =============================
class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        #To find the meaning of the actions, you should type env.get_action_meanings()
        self.gym_actions = range(gym_env.action_space.n)
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """
		#x_t1 is the screen
		#r_t is the reward
		#terminal is boolean indicating if the game is finished or not
		#info
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat, 84, 84))
        s_t1[:self.action_repeat-1, :] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


env = gym.make('MsPacman-v0')
env.reset()
c=AtariEnvironment(env,2)
s_t=c.get_initial_state()
#for _ in range(1000):
#	s_t1, r_t, terminal, info=c.step(env.action_space.sample())
	#print(r_t)
#	env.render()
#	#env.step(env.action_space.sample())
#	env.step(0)
s_t1, r_t, terminal, info=c.step(env.action_space.sample())
nactions=env.action_space.n
#model = Sequential()
#h = Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='same', activation='relu')(Input((1,84,84)))
#h = Convolution2D(32, 4, 4, subsample=(2, 2),border_mode='same', activation='relu')(h)
#h = Flatten()(h)
#h = Dense(256, activation='relu')(h)
#V = Dense(nactions)(h)
model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(4, 4),border_mode='same', activation='relu',input_shape=(1,84,84)))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

