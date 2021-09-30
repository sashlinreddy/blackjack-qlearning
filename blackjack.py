# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:41:36 2016

@author: Sashlin
"""
import numpy as np

cards = { "Ace"   : 1,
          "2"     : 2,
          "3"     : 3,
          "4"     : 4,
          "5"     : 5,
          "6"     : 6,
          "7"     : 7,
          "8"     : 8,
          "9"     : 9,
          "10"    : 10,
          "Jack"  : 10,
          "Queen" : 10,
          "King"  : 10 }
          
actions = { "Stay"  : 0,
            "Hit"   : 1 }

# A hand is represented as (faceValue, containsAce)
def getEmptyHand():
    return (0, False)

###############################################################################   
   
# Return whether or not a hand has a useable ace
def hasUseableAce(hand):
    faceValue, useableAce = hand
    return ((useableAce) and ((faceValue + 10) <= 21))

###############################################################################
    
def getHandValue(hand):
    faceValue = 0
    faceValue, useableAce = hand
    if hasUseableAce(hand):
        faceValue = faceValue + 10
    
    return faceValue
    
###############################################################################

# Update the hand and its face value
def addCardToHand(hand, card):
    faceValue, useableAce = hand
    faceValue = faceValue + card
    # If the card we are adding is an ace
    if card == 1:
        useableAce = True
        
    return (faceValue, useableAce)
    
###############################################################################

# Returns random card's face value    
def getRandomCard():    
    card = np.random.choice(list(cards.keys()))
    faceValue = list(cards.keys()).index(card)
    return faceValue
    
###############################################################################

# Deal the players hand. Add a new card until we achieve a value greater than 11
def dealPlayerHand():
    hand = getEmptyHand()
    hand = addCardToHand(hand, getRandomCard())
    hand = addCardToHand(hand, getRandomCard())
    while getHandValue(hand) < 11:
        hand = addCardToHand(hand, getRandomCard())
        
    return hand

###############################################################################    

def dealInitialDealerHand():
    hand = getEmptyHand()
    hand = addCardToHand(hand, getRandomCard())
    return hand

###############################################################################

def dealDealerHand(hand):
    
    while getHandValue(hand) < 17:
        hand = addCardToHand(hand, getRandomCard())
            
    return hand    

###############################################################################


# States are tuples (card, val, useable) where
#  - card is the card the dealer is showing
#  - val is the current value of the player's hand
#  - useable is whether or not the player has a useable ace

# Actions are either stay (False) or hit (True)

# Select a state at random.
def getRandomState(states):
   n = len(states)
   randomIndex = np.random.randint(0,n-1)
   state = states[randomIndex]
   return state
      
###############################################################################
      
def getInitialStates():
    states = []
    for dealerVal in np.arange(1, 11):
        for playerVal in np.arange(11, 21):
            playerUseableAce = True
            states.append((playerVal, dealerVal, playerUseableAce))
            states.append((playerVal, dealerVal, not playerUseableAce))            
    return states
    
###############################################################################

# A table of action values indexed by state and action. Initially zero
def setUpQTable():
    states = getInitialStates()
    Q = {}
    for state in states:
        Q[(state,actions['Stay'])] = 0.0
        Q[(state,actions['Hit'])] = 0.0
    return Q
    
###############################################################################

# Sets up a table of frequencies for state-action pairs. Initally zero    
def setUpStateActionFrequencyTable():
    NstateAction = setUpQTable()
    return NstateAction

###############################################################################

# Given the state, return player and dealer hand consistent with it.
def getPlayerAndDealerHandsFromState(state):
   playerVal, dealerCard, useableAce = state
   if (useableAce):
      playerVal = playerVal - 10
   playerHand = (playerVal, useableAce)
   dealerHand = getEmptyHand()
   dealerHand = addCardToHand(dealerHand, dealerCard)
   return dealerCard, dealerHand, playerHand
   
###############################################################################
    
# Given the dealer's card and player's hand, return the state.
def getStateFromDealerCardAndPlayerHand(dealerCard, playerHand):
   playerVal = getHandValue(playerHand)
   useableAce = hasUseableAce(playerHand)
   return (playerVal, dealerCard, useableAce)
   
###############################################################################
    
def discountFunction(NstateAction):
    return 1.0/NstateAction
    
###############################################################################

def calculateReward(playerHand, dealerHand):
    playerVal = getHandValue(playerHand)
    dealerVal = getHandValue(dealerHand)
    
    reward = 0.0
    # Check if player hand is better than dealer hand 
    if playerVal > dealerVal:
        reward = 1.0
    elif dealerVal > 21:
        reward = 1.0
    elif playerVal < dealerVal:
        reward = -1.0
    # Else playerVal == dealerVal and the reward remains 0.0
        
    return reward
    
###############################################################################

def qMax(Q, state):
    maxQVal = -1.0
    if Q[(state, actions['Hit'])] > Q[(state, actions['Stay'])]:
        maxQVal = Q[(state, actions['Hit'])]
    else:
        maxQVal = Q[(state, actions['Stay'])]
        
    return maxQVal

###############################################################################
    
def getGLIEpolicy(Q, state, epsilon):
    rand = np.random.random()
    if rand < epsilon:
        return getRandomAction()
    else:
        return getBestAction(Q, state)
        
###############################################################################
        
def getRandomAction():
    # print(actions.keys())
    actionKey = np.random.choice(list(actions.keys()))
    return actionKey
    
###############################################################################
    
def getBestAction(Q, state):
    if Q[state, actions['Hit'] > Q[state, actions['Stay']]]:
        return 'Hit'
    else:
        return 'Stay'
        
###############################################################################
        
def explorationFunction(Q, state, NstateAction, Nepsilon):
    if NstateAction < Nepsilon:
        return getBestAction(Q, state)
    else:
        return getRandomAction()
    
###############################################################################
    
def printPolicy(Q):
    
    print('\n---- Policy ----\n')
    for useable in [True, False]:  
        if useable:
            print('Soft Totals (Useable ace)')
        else:
            print('Hard Totals (No useable ace)')
        for i in np.arange(1, 11):
            print(i),
        print('\n')
        for val in np.arange(11,21):
            for card in np.arange(1,11):
                if (Q[((val,card,useable),actions['Hit'])] > Q[((val,card,useable),actions['Stay'])]):
                    print( 'H', end=',')
                else:
                    print( 'S', end=',')
            print('| %d' % val)
        print(' ')
            
##############################################################################

def runQLearning():
    delta = 1
    diff = 1e-6
    iteration = 1
    discount = 0.9
    
    # Setup Q table -> initially all zero
    Q = setUpQTable()
    # Setup state action freq table -> initially zero
    NstateAction = setUpStateActionFrequencyTable()
    # Setup states
    states = getInitialStates()
    
    while iteration < 100000:
        global Q_copy
        Q_copy = Q.copy()
        
        restart = False
        # get random state to start from 
        state = getRandomState(states)
        
        dealerCard, dealerHand, playerHand = getPlayerAndDealerHandsFromState(state)
        
        while not restart:
            
            # Choose random action from the dictionary
            # actionKey = np.random.choice(actions.keys())
            actionKey = getGLIEpolicy(Q, state, 0.7)
            # actionKey = explorationFunction(Q, state, NstateAction[state, ])
            action = actions[actionKey]

            # If the action is hit then we add a random card to the players hand            
            if actionKey == 'Hit':
                playerHand = addCardToHand(playerHand, getRandomCard())
                
                # Check if player has busted
                if getHandValue(playerHand) > 21:
                    # Increment state-action pair in freq table
                    NstateAction[(state, action)] = NstateAction[(state, action)] + 1.0
                    # Update Q-table
                    Q[state, action] = Q[state, action] + discountFunction(NstateAction[(state, action)]) \
                                       *((-1.0) + 0.0 - Q[state, action])
                    restart = True
                    print('Busted')
                    break
                elif getHandValue(playerHand) == 21:
                    # Increment state-action pair in freq table
                    NstateAction[(state, action)] = NstateAction[(state, action)] + 1.0
                    # Update Q-table
                    Q[state, action] = Q[state, action] + discountFunction(NstateAction[(state, action)]) \
                                       *((1.0) + 0.0 - Q[state, action])
                    restart = True
                    print('Yay')
                    break
                else:
                    # Increment state-action pair in freq table
                    NstateAction[(state, action)] = NstateAction[(state, action)] + 1.0
                    # Find qMax for the next state
                    nextState = getStateFromDealerCardAndPlayerHand(dealerCard, playerHand)
                    qMaxVal = qMax(Q, nextState)
                    # Update Q-table
                    Q[state, action] = Q[state, action] + discountFunction(NstateAction[(state, action)]) \
                                       *(0.0 + discount * qMaxVal - Q[state, action])
                    # Update state
                    state = nextState
                    print('Next State')
                    
            # Allow the dealer to play
            else:
                dealerHand = dealDealerHand(dealerHand)
                # Calculate immediate reward
                reward = calculateReward(playerHand, dealerHand)
                # Increment state-action pair in freq table
                NstateAction[(state, action)] = NstateAction[(state, action)] + 1.0
                # Update Q-table
                # qMax will be zero because this will be the terminal state
                Q[state, action] = Q[state, action] + discountFunction(NstateAction[(state, action)]) \
                                       *(reward + 0.0 - Q[state, action])
                restart = True
                break
                
        delta = np.max( np.abs( np.array(sorted(Q.values()))-np.array(sorted(Q_copy.values())) ) )
        print( 'iteration = ' + str(iteration) + ', ' + 'delta = ' + '%7.4f'%(delta) )
        iteration = iteration + 1
        
    return Q

Q = runQLearning()
                
printPolicy(Q)