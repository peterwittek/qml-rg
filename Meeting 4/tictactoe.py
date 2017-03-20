'''This is a template with a randomly behaving agent playing tic-tac-toe against
a perfect AI, that is, the best the agent can hope for is a tie.
To make it work, you will xo (https://github.com/dwayne/xo-python/). Install
it with pip, but *strictly* not with conda (that is a different xo):

pip install xo

It is written in pure Python, so you should not have any issues installing it.
'''

import random
from xo import ai
from xo.game import Game


def get_free_locations(game):
    free_locations = []
    for i, cell in enumerate(game.board.cells):
        if cell == ' ':
            free_locations.append((i // 3 + 1, i % 3 + 1))
    return free_locations


class RandomAgent(object):

    def __init__(self, game):
        self.game = game

    def next_move(self):
        free_locations = get_free_locations(self.game)
        return free_locations[random.randint(0, len(free_locations))]


# This is one round, ending with a reward
game = Game()
agent = RandomAgent(game)
game.start('o')
while True:
    ai_move = ai.evaluate(game.board, 'o').positions[0]
    status = game.moveto(*ai_move)
    print(game.board.toascii())
    if status['name'] == 'gameover':
        break
    agent_move = agent.next_move()
    status = game.moveto(*agent_move)
    print(game.board.toascii())
    if status['name'] == 'gameover':
        break
if game.statistics['xwins'] == 1:
    agent_reward = 1
elif game.statistics['squashed'] == 1:
    agent_reward = 0
else:
    agent_reward = -1
