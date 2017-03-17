
''' This agent is based on the Bellman equation, used to create a matrix which will help the agent to find the
suitable move, given a certain position. The learning function constructs the matrix from random games. This matrix has
an element for each reward position, agent position and move. There is an option at the end of the program to let the initial
position of the agent not be only the center, to make things trickier! '''

import numpy as np
import random


Q = np.zeros(shape = (2,3,2))  # We shape the Q matrix from Bellman equation for 2 reward states, 3 agent states, 2 actions
Agent_state = [[1,0,0],[0,1,0],[0,0,1]]  #possible agent positions

# An easy way to save the different states of the game is as follows: the reward can only have two
# positions, so we assign 0 for left and 1 for right. For the agent, we can have three different
# positions so 0 = left, 1 = center, 2 = right. Putting the reward + agent together, we can have 6
# possible states, that we can number from 0 to 5 using S=position_agent+3*position_reward. Then
# we can build a reward matrix, where for the states where there is a coincidence (S=0+0=0 and
# S=2+3*1=5) we assing value one, and for the non coinciding states we assing 0:  
             
Reward=[1, 0, 0, 0, 0, 1] # Reward matrix from Bellman equation. 

state = [0,0,0] # empty state

        

# Copy paste from rl_toy_example.py + comments from Patrick-------------------- 

def moveto(position,agent_position, status):
    agent_position += position  # x += y adds y to the variable x
    if agent_position < 0:  #since we always add 1 or -1 randomly to the position it can happen, that we get negative indices
        agent_position = 0 #position -1 therefore is always set to 0
        #agent_position = self.agent_position if we set this it will be found in max 2 steps  (CHEAT)
    elif agent_position > 2: #same for to big values. indices bigger than 2 are not in the state vector anymore
        agent_position = 2
    if status == "running" and state[agent_position] == 1: #if the agent is positioned at the same index as we initially set the reward
        status = "win"
        return 1, status, agent_position
    else:
        return 0, status, agent_position

def next_move():
    return random.randint(0, 1)*2-1 #gives randomly either 1 or -1
    
# ---------------------------------------------------------------------------- 



# Learning function ----------------------------------------------------------

def learning(idx_reward,agent_position,agent_move,gamma): # This function construct the matrix Q by means of the Bellman equation          
    position_agent_0 = agent_position # Position of the Agent at time t
    position_agent_1 = moveto(agent_move, agent_position, status)[2] # Position of the Agent at time t+1
    
    if agent_move == -1: # we assing the value 0 for a left move (agent_move=-1) and 1 for a right move
        a_move = 0
    else:
        a_move = 1
        
    if idx_reward == 2: # we assign 1 to the right position for the reward (idx_reward=2) and 0 for left position
        n_idx_reward = 1
    else:
        n_idx_reward = 0
    
    # Bellman equation:     
    Q[n_idx_reward, position_agent_0, a_move] = Reward[position_agent_1 + 3*n_idx_reward] + gamma*np.max([Q[n_idx_reward, position_agent_1, i] for i in [0,1]])
    
# ----------------------------------------------------------------------------- 



# Teached agent ---------------------------------------------------------------

def cleverplayer(idx_reward,agent_position):
    
    if idx_reward == 2: # we assign 1 to the right position for the reward (idx_reward=2) and 0 for left position
        n_idx_reward = 1
    else:
        n_idx_reward = 0    
    
    # We compare the value of Q for the given state and the two possible movements. The highest value
    # gives us the move to do! In the (rare) case the values are equal, we choose randomly
    if Q[n_idx_reward, agent_position, 0] > Q[n_idx_reward, agent_position, 1]:
        return +1
    elif Q[n_idx_reward, agent_position, 0] < Q[n_idx_reward, agent_position, 1]:
        return -1
    else:
        return random.randint(0, 1)*2-1

# -----------------------------------------------------------------------------
      


# Learning phase: the goal is to construct the matrix Q by playing random games 

iterations=100 # Number of iterations of the learning process
gamma=0.9 # gamma of the Bellman equation. Roughly, this parameter tells you how much you want to take in account from future moves

for ii in range(0,iterations):
 idx_reward=random.randint(0, 1)*2
 state[idx_reward] = 1    #places reward randomly in state
 agent_position = 1  
 agent_position = random.randint(0, 2) # Uncomment for random initial position (harder for the agent)
 status = "running"

 while status == "running":
    agent_move = next_move()
    learning(idx_reward, agent_position, agent_move,gamma)
    reward, status, agent_position = moveto(agent_move, agent_position, status) 
 
# -----------------------------------------------------------------------------
 



# Play! -----------------------------------------------------------------------

iterations_play=100
number_steps = 0
 
for jj in range(0,iterations_play):
 idx_reward=random.randint(0, 1)*2
 state[idx_reward] = 1    
 agent_position = 1  
 agent_position = random.randint(0, 2) # Uncomment for random initial position (harder for the agent)
 status = "running"

 
 while status == "running":
     number_steps += 1 #saving number of steps
     agent_move = cleverplayer(idx_reward,agent_position)
     reward, status, agent_position = moveto(agent_move, agent_position, status)

print('\n The mean number of steps is ' + repr(number_steps / iterations_play) + '\n')
     
     
     
     
     
     
     
     
     
     
