
'''This is a 'Braingent' wich will use a small neural network with keras and tensorflow as backend to learn the Q-function.
 it's obviously way to complicated for this game but it works very well and it should be possible to adapt it to play in AiGym etc.
At the end it contains a little sample code comparing the average number of moves the Braingent with Peters random agent. Since it saves the network weights, if you run
the code multiple times the Braingent will get better and better. for me it quickly reaches a mean of 1+e, where e is the exploration factor , whith which you choose a random move 
in Deep Q learning. At last the Braingent is run with exploration turned off, e=0, which gives an average step number of 1.0
'''
import random
import numpy as np
from keras.models import Model 
from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

import tensorflow as tf


class Game(object):

	def __init__(self):
		self.state = [0, 0, 0]
		self.agent_position = 1
		self.victim_position = random.randint(0, 1)*2
		self.state[self.victim_position] = 1
		self.status = "running"

	def moveto(self, position):
		self.agent_position += position
		if self.agent_position < 0:
			self.agent_position = 0
		elif self.agent_position > 2:
			self.agent_position = 2
		if self.status == "running" and self.state[self.agent_position] == 1:
			self.status = "gameover"
			return 1
		else:
			return 0
	def stat(self):
		return [self.agent_position, self.victim_position]

	def reward(self):
		if self.agent_position == self.victim_position:
			return 1
		else:
			return 0
	


class RandomAgent(object):

	def __init__(self, game):
		self.game = game
		self.number_of_steps = 0

	def next_move(self):
		self.number_of_steps += 1
		return random.randint(0, 1)*2-1

class Braingent:

	def __init__(self,knowhow="brain.h5" ,cap=100,epsilon=0.1,gamma=0.9):
		self.knowhow=knowhow
		self.memory = []
		self.gamma=gamma
		self.cap = cap
		self.epsilon=epsilon
		self.build_brain()

	def build_brain(self):
		inp = Input(shape=(2,))
		l1 = Dense(3,activation='relu')(inp)
		out = Dense(2,activation='linear')(l1)
		self.brain=Model(inp,out)
		self.brain.compile(loss='mean_squared_error', optimizer='sgd')
		
		Ql = Lambda(lambda x: tf.reduce_max(x,axis=[1],keep_dims=False))(self.brain(inp))
		self.Q = Model(inp,Ql)	

		Al = Lambda(lambda x: tf.argmax(x,axis=1))(self.brain(inp))
		self.A = Model(inp,Al)
		try:
			self.brain.load_weights(self.knowhow)
		except:
			print('new braingent, no knowhow')

	def estimateReward(self,states,rewards):
		return	self.gamma*self.Q.predict(states) + rewards


	def chooseAction(self,state):
		if np.random.random() < self.epsilon:
			a = random.randint(-1,1)
		else:
			a= self.A.predict(state)[0]
			if a == 0.0:
				a = -1
			else:
				a = 1
		return a

	def new_game(self,game):
		self.game = game
		self.i = 0
		self.memory.append([])
		self.memory= self.memory[-self.cap:]

	def act(self):
		state = np.asarray([self.game.stat()],dtype='float32')
		reward =np.asarray([self.game.reward()],dtype='float32')
		action =  self.chooseAction(state)
		self.memory[-1].append([game.stat(),game.reward(),action])
		game.moveto(action)
		self.i += 1

	def learn(self,batch_size):
		n_games= len(self.memory)
		X = []
		Y = []
		for i in range(batch_size-1):
			if n_games == 1:
				gm = 0
			else:
				gm = random.randint(0,n_games-1)
			n_moves = len(self.memory[gm])
			if n_moves == 1:
				mv = 0
			else:
				mv = random.randint(0,n_moves-1)
			
			X.append(self.memory[gm][mv][0])
			if mv < n_moves-1:
				state_mv1 = np.asarray([self.memory[gm][mv+1][0]],dtype='float32')
				rew = np.asarray([self.memory[gm][mv][1]],dtype='float32')
				target = self.estimateReward(state_mv1,rew)[0]
				Y.append([target,target])
			else:
				target = self.memory[gm][mv][1]						   
				Y.append([target,target])
		X = np.asarray(X,dtype='float32')
		Y = np.asarray(Y,dtype='float32')
		self.brain.train_on_batch(X,Y)
		self.brain.save_weights(self.knowhow)
		
		
		


bragent= Braingent(epsilon=0.01)




timelapse=[]

for i in range(500):
	#print(i)
	game = Game()
	agent= RandomAgent(game)
	while game.status == 'running':
		game.moveto(agent.next_move())
	timelapse.append(agent.number_of_steps)

print('average number of moves per game for random agent:')
print(np.array(timelapse).mean())

timelapse=[]

for i in range(500):
	#print(i)
	game = Game()
	bragent.new_game(game)
	while game.status == 'running':
		bragent.act()
		bragent.learn(20)
	timelapse.append(bragent.i)


print('average number of moves per game for braingent:')
print(np.array(timelapse).mean())

timelapse=[]
bragent.epsilon=0

for i in range(500):
	#print(i)
	game = Game()
	bragent.new_game(game)
	while game.status == 'running':
		bragent.act()
		bragent.learn(20)
	timelapse.append(bragent.i)

print('average number of moves per game for trained braingent without exploration:')
print(np.array(timelapse).mean())

