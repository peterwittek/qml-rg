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
import qutip as qt


"""
----------.... -----
| 1 | 2 | .... | n | states
----------.... -----
"""

class Gamer():
    
    def __init__(self, n): #n will be the number of sites 
        self.states = [qt.basis(n,i) for i in range(0,n)] #generate list with "Fock" states from 1 to n      
        self.state = self.states[random.randint(0,n-1)] #initial state of invador is set randomly
        
        #Implement Grover
        self.s = 1/np.sqrt(n)*np.sum(self.states)
        self.Iwo = qt.qeye(n) - 2*self.state*self.state.dag()
        self.Is = qt.qeye(n) - 2*self.s*self.s.dag()
        self.grover = -self.Is*self.Iwo

        
        
dim = 3  
     
game = Gamer(dim)


for i in range(1,100): #number of iterations 
    a = game.grover**i*game.s
    print(a.overlap(game.state)) #calc overlapp between iterated state a and target state game.state

 

