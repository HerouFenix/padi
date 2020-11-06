#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model
# 
# Consider once again the "Doom" domain, described in the Homework which you modeled using a Markov decision process. In this environment, 
# 
# * The agent can move in any of the four directions: up, down, left, and right. 
# * Movement across a grey cell division succeeds with a 0.8 probability and fails with a 0.2 probability. 
# * Movements across colored cell divisions (blue or red) succeed with a 0.8 probability (and fail with a probability 0.2) only if the agent has the corresponding colored key. Otherwise, they fail with probability 1. To get a colored key, the agent simply needs to stand in the corresponding cell.
# * When the movement fails, the agent remains in the same cell. 
# * The action that takes the agent through the exit always succeeds.  
# 
# In this lab you will interact with larger version of the same problem. You will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="maze.png" width="400px">
# 
# We consider that the agent is never in a cell $c\geq 17$ without a red key, and is never in a cell $c\geq28$ without a blue key. **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `X` that contains all the states in the MDP. There is a total of 73 states describing the possible positions of the agent in the environment and whether or not the agent has each of the two keys. Those states are represented as strings taking one of the forms `"N"`, indicating that the agent is in cell `N`, `"NR"`, indicating that the agent is in cell `N` with the red key, `"NRB"`, indicating that the agent is in cell `N` with both keys, or `"E"`, indicating that the agent has reached the exit.
# * An array `A` that contains all the actions in the MDP. Each action is represented as a string `"u"`, `"d"`, `"l"` or `"r"`.
# * An array `P` containing 4 $73\times 73$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., 3), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 4 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[1]:


import numpy as np

def load_mdp(file_name, gama):
    mdp = []
    with np.load(file_name) as data:
        # X
        x = data['X.npy']
        mdp.append(x)
        
        # A
        a = data['A.npy']
        mdp.append(a)
        
        # P
        p = tuple(data['P.npy'])
        mdp.append(p)
        
        # c
        c = data['c.npy']
        mdp.append(c)
        
        #g
        mdp.append(gama)
        
    return tuple(mdp)


import numpy.random as rand

M = load_mdp('maze.npz', 0.9)

rand.seed(42)

# States
print('Number of states:', len(M[0]))

# Random state
s = rand.randint(len(M[0]))
print('Random state:', M[0][s])

# Final state
print('Final state:', M[0][-1])

# Actions
print('Number of actions:', len(M[1]))

# Random action
a = rand.randint(len(M[1]))
print('Random action:', M[1][a])

# Transition probabilities
print('Transition probabilities for the selected state/action:')
print(M[2][a][s, :])

# Cost
print('Cost for the selected state/action:')
print(M[3][s, a])

# Discount
print('Discount:', M[4])


# We provide below an example of application of the function with the file `maze.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('maze.npz', 0.9)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0][s])
# 
# # Final state
# print('Final state:', M[0][-1])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[2][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[3][s, a])
# 
# # Discount
# print('Discount:', M[4])
# ```
# 
# Output:
# 
# ```
# Number of states: 73
# Random state: 9RB
# Final state: E
# Number of actions: 4
# Random action: u
# Transition probabilities for the selected state/action:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0.]
# Cost for the selected state/action:
# 1.0
# Discount: 0.9
# ```

# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# Write a function `noisy_policy` that builds a noisy policy "around" a provided action. Your function should receive, as input, an MDP described as a tuple like that of **Activity 1**, an integer `a`, corresponding to the index of an action in the MDP, and a real number `eps`. The function should return, as output, a policy for the provided MDP that selects action with index `a` with a probability `1-eps` and, with probability `eps`, selects another action uniformly at random. The policy should be a `numpy` array with as many rows as states and as many columns as actions, where the element in position `[s,a]` should contain the probability of action `a` in state `s` according to the desired policy. 
# 
# ---

# In[2]:


def noisy_policy(mdp, a, eps):
    policy = np.zeros((len(mdp[0]), len(mdp[1])))
    
    prob_other_actions = eps/(len(mdp[1])-1)
    
    for row in range(policy.shape[0]):
        for col in range(policy.shape[1]):
            if(col == a):
                policy[row][col] = 1 - eps
            else:
                policy[row][col] = prob_other_actions
    
    return policy


# Noiseless policy for action "Left" (action index: 2)
pol_noiseless = noisy_policy(M, 2, 0.)

# Random state
s = rand.randint(len(M[0]))

# Policy at selected state
print('Random state:', M[0][s])
print('Noiseless policy at selected state:', pol_noiseless[s, :])

# Noisy policy for action "Left" (action index: 2)
pol_noisy = noisy_policy(M, 2, 0.1)

# Policy at selected state
print('Noisy policy at selected state:', pol_noisy[s, :])

# Random policy for action "Left" (action index: 2)
pol_random = noisy_policy(M, 2, 0.75)

# Policy at selected state
print('Random policy at selected state:', pol_random[s, :])


# We provide below an example of application of the function with MDP from the example in **Activity 1**, that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ. Note also that your function should work with **any** MDP that is specified as a tuple like the one from **Activity 1**.
# 
# ```python
# # Noiseless policy for action "Left" (action index: 2)
# pol_noiseless = noisy_policy(M, 2, 0.)
# 
# # Random state
# s = rand.randint(len(M[0]))
# 
# # Policy at selected state
# print('Random state:', M[0][s])
# print('Noiseless policy at selected state:', pol_noiseless[s, :])
# 
# # Noisy policy for action "Left" (action index: 2)
# pol_noisy = noisy_policy(M, 2, 0.1)
# 
# # Policy at selected state
# print('Noisy policy at selected state:', pol_noisy[s, :])
# 
# # Random policy for action "Left" (action index: 2)
# pol_random = noisy_policy(M, 2, 0.75)
# 
# # Policy at selected state
# print('Random policy at selected state:', pol_random[s, :])
# ```
# 
# Output:
# 
# ```
# Random state: 15
# Noiseless policy at selected state: [0. 0. 1. 0.]
# Noisy policy at selected state: [0.03 0.03 0.9  0.03]
# Random policy at selected state: [0.25 0.25 0.25 0.25]
# ```

# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# ---

# In[3]:


def evaluate_pol(mdp,policy):
    Jpi = np.zeros((len(mdp[0]), 1))
    
    cPi = np.sum(policy * mdp[3], axis=1)

    Ppi = np.zeros((len(mdp[0]), len(mdp[0])))
    for a in range(len(mdp[1])):
        Ppi += np.diag(policy[:,a]).dot(mdp[2][a])
    
    #Jpi = (I-gama * Ppi)^-1 * cPi
    Jpi = np.linalg.inv(np.identity(len(mdp[0])) - mdp[4] * Ppi).dot(cPi)
    
    return Jpi.reshape((len(mdp[0]),1))
'''

def evaluate_pol(mdp, policy):
    cPi = np.zeros((len(mdp[0]),1))
    for i in range(0,len(mdp[1])):
        cAction = mdp[3][:,i].reshape(len(mdp[0]),1)
        cPi += np.diag(policy[:, i]).dot(cAction)
        
    pPi = np.zeros((len(mdp[0]),len(mdp[0])))
    for i in range(0,len(mdp[1])):
        pPi += np.diag(policy[:, i]).dot(mdp[2][i])
    
    
    return np.linalg.inv(np.eye(len(mdp[0])) - mdp[4] * pPi).dot(cPi)
'''
 
Jpi = evaluate_pol(M, pol_noisy)

rand.seed(42)

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])


# 
# As an example, you can evaluate the random policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jpi = evaluate_pol(M, pol_noisy)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# ```
# 
# Output: 
# ```
# Cost to go at state 9RB: [10.]
# Cost to go at state 15: [10.]
# Cost to go at state 29RB: [9.6]
# ```

# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# ---

# In[4]:


import time

def value_iteration(mdp):
    iterations = 0
    start_time = time.time()
    
    
    gamma = mdp[4]
    J = np.zeros((len(mdp[0]), 1))
    err = 1
    
    while err > 1e-8:
        
        cAction = mdp[3][:,0].reshape(len(mdp[0]),1)
        Qmin = cAction + gamma * mdp[2][0].dot(J)

        for i in range(1, len(mdp[1])):
            cAction = mdp[3][:,i].reshape(len(mdp[0]),1)
            Qnew = cAction + gamma * mdp[2][i].dot(J)
            
            Qmin = np.min((Qnew, Qmin), axis=0)
        
        Jnew = Qmin
        err = np.linalg.norm(Jnew-J)
        iterations+=1
        
        J = Jnew
    
    total_time = time.time() - start_time
    print("Execution time: {:.3f} seconds".format(total_time))
    print("N. iterations: {}".format(iterations))
        
    return J

    
Jopt = value_iteration(M)

rand.seed(42)

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))


# For example, the optimal cost-to-go for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# Jopt = value_iteration(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.006 seconds
# N. iterations: 73
# Cost to go at state 9RB: [6.82]
# Cost to go at state 15: [9.79]
# Cost to go at state 29RB: [1.]
# 
# Is the policy from Activity 2 optimal? False
# ```

# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Your function should print the time it takes to run before returning, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[5]:


def policy_iteration(mdp):
    iterations = 0
    start_time = time.time()
    
    gamma = mdp[4]
    pol = np.ones((len(mdp[0]), len(mdp[1])))/len(mdp[1])
    done = False
    
    while not done:
        J = evaluate_pol(mdp,pol)
        
        # Q function
        QArr = []
        for i in range(0, len(mdp[1])):
            cAction = mdp[3][:,i].reshape(len(mdp[0]),1)
            QAction = cAction + gamma * mdp[2][i].dot(J)
            QArr.append(QAction)

        QMin = np.min(QArr, axis = 0)
                            
        polNew = np.zeros((len(mdp[0]),len(mdp[1])))
        
        for i in range(0, len(mdp[1])):
            polNew[:, i, None] = np.isclose(QArr[i], QMin, atol=1e-8, rtol=1e-8).astype(int)
        
        polNew = polNew / np.sum(polNew, axis=1, keepdims = True)
        
        done = (pol == polNew).all()
        pol = polNew
        iterations += 1
    
    
    total_time = time.time() - start_time
    print("Execution time: {:.3f} seconds".format(total_time))
    print("N. iterations: {}".format(iterations))

    return pol


popt = policy_iteration(M)

rand.seed(42)

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))


# For example, the optimal policy for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# popt = policy_iteration(M)
# 
# rand.seed(42)
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# ```
# 
# Output:
# ```
# Execution time: 0.005 seconds
# N. iterations: 3
# Policy at state 9RB: l
# Policy at state 29RB: r
# Policy at state 8R: u
# ```

# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, `x0`, corresponding to a state index
# * A second integer, `length`
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **`NRUNS`** trajectories of `length` steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---

# In[15]:


NRUNS = 100

def simulate(mdp, policy, x0, length):
    total_cost = 0
    
    for run in range(NRUNS):        
        #Generate Trajectory
        trajectory = [x0]
        accumulated_cost = 0
                
        curState = x0
        curGamma = 1
        for i in range(length):
            # 1. Pick which action
            action = np.random.choice(range(0,len(mdp[1])), p=policy[curState,:])

            # 2. Choose which state to go to using picked action
            curDist = np.zeros((1,len(mdp[0])))
            curDist[0][curState] = 1
            distributions = np.matmul(curDist, mdp[2][action])

            state = np.random.choice(range(0,len(mdp[0])), p=distributions[0])
            
            trajectory.append(state)
            
            curState = state
            
            # Compute accumulated cost
            accumulated_cost += curGamma * mdp[3][curState][action]
            curGamma = curGamma * mdp[4]
        
        total_cost += accumulated_cost
        
    #Compute Average Cost
    average_cost = total_cost/NRUNS
    
    return average_cost
    
rand.seed(42)

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s, 100))

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s, 1000))

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s, 10000))


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 100))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 1000))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 10000))
# ```
# 
# Output:
# ````
# Cost-to-go for state 9RB:
# 	Theoretical: [6.82]
# 	Empirical: 6.864862326263111
# Cost-to-go for state 4:
# 	Theoretical: [9.94]
# 	Empirical: 9.944922179980777
# Cost-to-go for state 5:
# 	Theoretical: [9.94]
# 	Empirical: 9.937132279574923
# ```

# Perguntar:
#     Act 4. round up always ou ta bem como esta?
