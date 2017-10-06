'''This is a toy reinforcement learning problem. The state space has three
locations on a line, the agent starts in the middle one. Either on the left
or the right, there is a reward. Reaching this, the game terminates. The agent
has complete access to the state space of the game. The action space of the
agent is left or right moves. It can choose left infinite many times if the
reward is on the right: in this case, it will stay in the left-most cell
forever.
'''

import random


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

game = Game()
agent = RandomAgent(game)
while game.status == "running":
    agent_move = agent.next_move()
    game.moveto(agent_move)

print(agent.number_of_steps)
