import numpy as np
import random


Q = np.zeros(shape = (2,3,2))                       #shape for 2 reward states, 3 agent states, 2 actions
Agent_state = [[1,0,0],[0,1,0],[0,0,1]]             #possible agent positions
Reward = [[1,0,0],[0,0,1]]                          #possible reward positions
iterations = 100                                #how many times do we play the game




def moveto(position,agent_position, status):
    agent_position += position                                          # x += y adds y to the variable x
    if agent_position < 0:                                      #since we always add 1 or -1 randomly to the position it can happen, that we get negative indices
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
    
def next_move_intelligent(gamma):
    R = Reward.index(state)
    A = agent_position
    action = [-1,1]
    ind = [0,1]
    
    
    test = [[Q[R][A][i] + gamma*Q[R][A+action[i]][j] for j in ind] for i in ind]

    a = test.index(max(test)) #find maximum value in test and check which index it is. This will then be the next action
    
    return action[a]
    
#----------------------------------------------------------------------------------------------------------
#learning phase
#----------------------------------------------------------------------------------------------------------
    
for i in range(0,iterations):
    i += 1
    move_memory = []                     #initialize a memory for the moves for every new game
    state = [0,0,0]                      #empty the state for every new game
    state[random.randint(0, 1)*2] = 1    #places reward randomly in state
    agent_position = 1                   #initializes agent in the middle
    status = "running"
    move_memory.append([Reward.index(state), agent_position, 'initialize',0])

    while status == "running":
        agent_move = next_move()
        
        if agent_move < 0:
            action = "left"
        else:
            action = "right"    
        
        reward, status, agent_position=moveto(agent_move, agent_position, status) #the argument of moveto is either 1 or -1 and is the value added to the position
        
        index_reward = Reward.index(state) #which vector in Reward is actually the reward position
        index_Agent = agent_position

        
        move_memory.append([index_reward,index_Agent,action, reward])
        #print(move_memory)

            
    length = len(move_memory)

    for i in range(1,length):   #CONSTRUCT Q MATRIX
        k = move_memory[i-1][0]   #reward index
        l = move_memory[i-1][1]   #agent index
        print(l)
        if action == "left": #change left / right to index
            m= 0
        else:
            m= 1

        Q[k][l][m]=move_memory[i][3]

#-------------------------------------------------------------------------------------
#Let Agent play now 
   
state = [0,0,0]                      #empty the state for every new game
state[random.randint(0, 1)*2] = 1    #places reward randomly in state
agent_position = 1                   #initializes agent in the middle
status = "running"
moves = 0
move_memory = []


while status == "running":
    moves += 1
    agent_move = next_move_intelligent(0.9)
    
    if agent_move < 0:
        action = "left"
    else:
        action = "right"    
    
    reward, status, agent_position=moveto(agent_move, agent_position, status) #the argument of moveto is either 1 or -1 and is the value added to the position
    
    index_reward = Reward.index(state) #which vector in Reward is actually the reward position
    index_Agent = agent_position

    
    move_memory.append([index_reward,index_Agent,action, reward])
    print(move_memory)
    
print(moves)
   

    
    
