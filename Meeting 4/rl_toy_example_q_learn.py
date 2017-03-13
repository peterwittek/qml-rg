'''This is a toy reinforcement learning problem. The state space has three
locations on a line, the agent starts in the middle one. Either on the left
or the right, there is a reward. Reaching this, the game terminates. The agent
has complete access to the state space of the game. The action space of the
agent is left or right moves. It can choose left infinite many times if the
reward is on the right: in this case, it will stay in the left-most cell
forever.
'''

import random
import numpy as np


class Game(object):

    def __init__(self):
        self.state = [0, 0, 0]
        self.agent_position = 1
        self.state[random.randint(0, 1)*2] = 1
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

class RandomAgent(object):

    def __init__(self, game):
        self.game = game
        self.number_of_steps = 0

    def next_move(self):
        self.number_of_steps += 1
        return random.randint(0, 1)*2-1

class QAgent(object):
    
    def __init__(self, game):
        self.game = game
        self.number_of_steps = 0
        self.Q = np.ones([2,3,2])#np.random.uniform(size = [2,3,2])
   
    def next_move(self, state):
        self.number_of_steps += 1
        return 2*np.argmax(self.Q[state, self.game.agent_position, :])-1
        
    def update_Q(self, old_position, move, new_position, reward):
        learn_rate, discount = 1.0, 0.5
        self.Q[state, old_position, move] += learn_rate*(reward + 
              discount*np.max(self.Q[state, new_position, :]) - 
                             self.Q[state, old_position, move])
                     
game = Game()
agent = RandomAgent(game)
while game.status == "running":
    agent_move = agent.next_move()
    game.moveto(agent_move)

print(agent.number_of_steps)

game_Q = Game()
agent_Q = QAgent(game_Q)

for i in range(0, 10):
    game_Q = Game()
    agent_Q.game = game_Q
    agent_Q.number_of_steps = 0
    state = int(game_Q.state == [0, 0, 1])
    while game_Q.status == "running":
        agent_move = agent_Q.next_move(state)
        old_position = game_Q.agent_position
        reward = game_Q.moveto(agent_move)
        new_position = game_Q.agent_position
        agent_Q.update_Q(old_position, (agent_move + 1)//2, new_position, reward)
        
    print(agent_Q.number_of_steps)