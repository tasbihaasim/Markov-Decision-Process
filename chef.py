import numpy as np

# Define the MDP states and actions
states = []
actions = [0,1,2,3]
rewards = {}

L=0
R=1
U=2
D=3


def make_rewards(st):
    '''Takes the state and calculate rewards for the 4 actions'''
    x = st[0]
    y = st[1]
    rewards[st] = {L:0, R:0, U:0, D:0}
    ## wall states
    if y==1:
        rewards[st][D] = -1
        if x==1: #2
            rewards[st][L] = -1
        if x==9:
            rewards[st][R] = -1
        if x==8 or x==1 or x==2: #4
            rewards[st][U] = -1
    if y==2:
        if x==1 or x==6 or x==8:
            rewards[st][L] = -1
        if x==4 or x==7 or x==9:
            rewards[st][R] = -1
        if x==3 or x==2:
            rewards[st][U] = -1
        if x==1 or x==2 or x==8:
            rewards[st][D] = -1
    if y==3:
        if x==1 or x==2 or x==6 or x==8:
            rewards[st][L] = -1
        if x==1 or x==4 or x==7 or x==9:
            rewards[st][R] = -1
        if x==1 or x==8 or x==9:
            rewards[st][U] = -1
        if x==2 or x==3:
            rewards[st][D] = -1
    if y==4:
        rewards[st][U] = -1
        if x==1 or x==6:
            rewards[st][L] = -1
        if x==4 or x==9:
            rewards[st][R] = -1
        if x==1 or x==8 or x==9:
            rewards[st][D] = -1
    ## gate states
    if x==4:
        rewards[st][R] == -1
        if y == [1]:
            rewards[st][R] == 20  ## should attempt to go to the right of the grid
    if x==6:
        rewards[st][L] == -1
        if y == [1]:
            rewards[st][L] == -20 ## should not attempt to go to the left grid
    ## egg beater states
    if x==1 and y==2 and st[2] == False: ## move up towards egg beater 2
        rewards[st][U] = 500
    if x==8 and y==2 and st[2] == False: ## move up towards egg beater 2
        rewards[st][U] = 500
    if x==9 and y==3 and st[2] == False: ## move left towards egg beater 2
        rewards[st][L] = 500
    ## goal states
    if x==7 and y==4 and st[2] == True: ## move right towards pudding maker
        rewards[st][R] = 1000
    if x==9 and y==4 and st[2] == True: ## move left towards pudding maker
        rewards[st][L] = 1000


##create the states array
for y in range(1,5):
    for x in range(1,5):
        states.append((x, y, False)) #when no egg beater 
        states.append((x, y, True)) #when yes egg beater
    for x in range(6, 10):
        states.append((x, y, False)) #when no egg beater 
        states.append((x, y, True)) #when yes egg beater

## assign rewards for each state
for i in states:
    make_rewards(i)

num_states = len(states)
num_actions = len(actions)

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))




#take action function
    
def take_action(s, action):
    state = states[s]
    next_state = state
    ## get the next state 
    ## move coordinates
    if action == 0 and state[0]>1:
        next_state = (state[0]-1, state[1], state[2])
        if state[0]==6:
            next_state = (state[0], state[1], state[2])
            if state[1] == 1:
                next_state = (state[0]-2, state[1], state[2])
    if action == 1 and state[0]<9:
        next_state = (state[0]+1, state[1], state[2])
        if state[0]==4:
            next_state = (state[0], state[1], state[2])
            if state[1] == 1:
                next_state = (state[0]+2, state[1], state[2])
    if action == 2 and state[1]<4:
        next_state = (state[0], state[1]+1, state[2])
    if action == 3 and state[1]>1:
        next_state = (state[0], state[1]-1, state[2])
    ## if the state goes from no egg beater to yes egg beater
    if state[0] == 8 and state[1]==2 and state[2]==False and action==2:
        next_state = (8,3, True)
    if state[0] == 9 and state[1]==3 and state[2]==False and action==0:
        next_state = (8,3, True)
    if state[0] == 1 and state[1]==2 and state[2]==False and action==2:
        next_state = (1,3, True)

    ## if wall state
    if rewards[state][action] == -1:
        done = True
        reward = -10000
        ns = states.index(next_state)
        return ns, reward, done
    ## if reward state
    elif rewards[state][action]==500:
        done = False
        reward = rewards[state][action]
        ns = states.index(next_state)
        return ns, reward, done
    elif rewards[state][action]==1000:
        done = True ## termination state
        reward = rewards[state][action]
        ns = states.index(next_state)
        return ns, reward, done
    ## if empty cell
    else:
        done = False
        ns = states.index(next_state)
        reward = rewards[state][action]
        return ns, reward, done
        

## the rows of Q-table is states. 

# Hyperparameters
alpha = 0.1 
gamma = 0.9 ## discount factor
epsilon = 1 ## learning rate

# Training loop
for episode in range(1000):
    # Initialize state
    if (episode%100)==0 and epsilon>=0.2:
        epsilon = epsilon - 0.1
        epsilon = round(epsilon,1)
    current_state = np.random.randint(num_states)
    while True:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[current_state])
        
        # Take action and observe reward and next state
        next_state, reward, done = take_action(current_state, action)
        # Update Q-table
        q_table[current_state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[current_state, action])

        # Update current state
        current_state = next_state
        # Check if episode is done
        if done:
            break


# Extract optimal policy from Q-table
print(q_table)
optimal_policy = np.argmax(q_table, axis=1)

## print the optimal policy for each state
for i in range(len(states)):
    direction = ""
    if optimal_policy[i] == 0:
        direction = "left"
    if optimal_policy[i] == 1:
        direction = "right"
    if optimal_policy[i] == 2:
        direction = "up"
    if optimal_policy[i] == 3:
        direction = "down"
    print(states[i], "---->", direction)



