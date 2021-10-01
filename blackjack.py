# -*- coding: utf-8 -*-
"""
created on sun oct 16 21:41:36 2016

@author: sashlin
"""
import numpy as np

cards = { "ace"   : 1,
          "2"     : 2,
          "3"     : 3,
          "4"     : 4,
          "5"     : 5,
          "6"     : 6,
          "7"     : 7,
          "8"     : 8,
          "9"     : 9,
          "10"    : 10,
          "jack"  : 10,
          "queen" : 10,
          "king"  : 10 }
          
actions = { "stay"  : 0,
            "hit"   : 1 }

# a hand is represented as (face_value, contains_ace)
def get_empty_hand():
    return (0, False)

###############################################################################   
   
# return whether or not a hand has a useable ace
def has_useable_ace(hand):
    face_value, useable_ace = hand
    return ((useable_ace) and ((face_value + 10) <= 21))

###############################################################################
    
def get_hand_value(hand):
    face_value = 0
    face_value, useable_ace = hand
    if has_useable_ace(hand):
        face_value = face_value + 10
    
    return face_value
    
###############################################################################

# update the hand and its face value
def add_card_to_hand(hand, card):
    face_value, useable_ace = hand
    face_value = face_value + card
    # if the card we are adding is an ace
    if card == 1:
        useable_ace = True
        
    return (face_value, useable_ace)
    
###############################################################################

# returns random card's face value    
def get_random_card():    
    card = np.random.choice(list(cards.keys()))
    face_value = list(cards.keys()).index(card)
    return face_value
    
###############################################################################

# deal the players hand. add a new card until we achieve a value greater than 11
def deal_player_hand():
    hand = get_empty_hand()
    hand = add_card_to_hand(hand, get_random_card())
    hand = add_card_to_hand(hand, get_random_card())
    while get_hand_value(hand) < 11:
        hand = add_card_to_hand(hand, get_random_card())
        
    return hand

###############################################################################    

def deal_initial_dealer_hand():
    hand = get_empty_hand()
    hand = add_card_to_hand(hand, get_random_card())
    return hand

###############################################################################

def deal_dealer_hand(hand):
    
    while get_hand_value(hand) < 17:
        hand = add_card_to_hand(hand, get_random_card())
            
    return hand    

###############################################################################


# states are tuples (card, val, useable) where
#  - card is the card the dealer is showing
#  - val is the current value of the player's hand
#  - useable is whether or not the player has a useable ace

# actions are either stay (False) or hit (True)

# select a state at random.
def get_random_state(states):
   n = len(states)
   random_index = np.random.randint(0,n-1)
   state = states[random_index]
   return state
      
###############################################################################
      
def get_initial_states():
    states = []
    for dealer_val in np.arange(1, 11):
        for player_val in np.arange(11, 21):
            player_useable_ace = True
            states.append((player_val, dealer_val, player_useable_ace))
            states.append((player_val, dealer_val, not player_useable_ace))            
    return states
    
###############################################################################

# a table of action values indexed by state and action. initially zero
def set_up_q_table():
    states = get_initial_states()
    q = {}
    for state in states:
        q[(state,actions['stay'])] = 0.0
        q[(state,actions['hit'])] = 0.0
    return q
    
###############################################################################

# sets up a table of frequencies for state-action pairs. initally zero    
def set_up_state_action_frequency_table():
    nstate_action = set_up_q_table()
    return nstate_action

###############################################################################

# given the state, return player and dealer hand consistent with it.
def get_player_and_dealer_hands_from_state(state):
   player_val, dealer_card, useable_ace = state
   if (useable_ace):
      player_val = player_val - 10
   player_hand = (player_val, useable_ace)
   dealer_hand = get_empty_hand()
   dealer_hand = add_card_to_hand(dealer_hand, dealer_card)
   return dealer_card, dealer_hand, player_hand
   
###############################################################################
    
# given the dealer's card and player's hand, return the state.
def get_state_from_dealer_card_and_player_hand(dealer_card, player_hand):
   player_val = get_hand_value(player_hand)
   useable_ace = has_useable_ace(player_hand)
   return (player_val, dealer_card, useable_ace)
   
###############################################################################
    
def discount_function(nstate_action):
    return 1.0/nstate_action
    
###############################################################################

def calculate_reward(player_hand, dealer_hand):
    player_val = get_hand_value(player_hand)
    dealer_val = get_hand_value(dealer_hand)
    
    reward = 0.0
    # check if player hand is better than dealer hand 
    if player_val > dealer_val:
        reward = 1.0
    elif dealer_val > 21:
        reward = 1.0
    elif player_val < dealer_val:
        reward = -1.0
    # else player_val == dealer_val and the reward remains 0.0
        
    return reward
    
###############################################################################

def q_max(q, state):
    max_q_val = -1.0
    if q[(state, actions['hit'])] > q[(state, actions['stay'])]:
        max_q_val = q[(state, actions['hit'])]
    else:
        max_q_val = q[(state, actions['stay'])]
        
    return max_q_val

###############################################################################
    
def get_gli_epolicy(q, state, epsilon):
    rand = np.random.random()
    if rand < epsilon:
        return get_random_action()
    else:
        return get_best_action(q, state)
        
###############################################################################
        
def get_random_action():
    # print(actions.keys())
    action_key = np.random.choice(list(actions.keys()))
    return action_key
    
###############################################################################
    
def get_best_action(q, state):
    if q[state, actions['hit'] > q[state, actions['stay']]]:
        return 'hit'
    else:
        return 'stay'
        
###############################################################################
        
def exploration_function(q, state, nstate_action, nepsilon):
    if nstate_action < nepsilon:
        return get_best_action(q, state)
    else:
        return get_random_action()
    
###############################################################################
    
def print_policy(q):
    
    print('\n---- Policy ----\n')
    for useable in [True, False]:  
        if useable:
            print('Soft totals (useable ace)')
        else:
            print('Hard totals (no useable ace)')
        for i in np.arange(1, 11):
            print(i),
        print('\n')
        for val in np.arange(11,21):
            for card in np.arange(1,11):
                if (q[((val,card,useable),actions['hit'])] > q[((val,card,useable),actions['stay'])]):
                    print( 'H', end=',')
                else:
                    print( 'S', end=',')
            print('| %d' % val)
        print(' ')
            
##############################################################################

def run_q_learning():
    delta = 1
    diff = 1e-6
    iteration = 1
    discount = 0.9
    
    # setup q table -> initially all zero
    q = set_up_q_table()
    # setup state action freq table -> initially zero
    nstate_action = set_up_state_action_frequency_table()
    # setup states
    states = get_initial_states()
    
    while iteration < 100000:
        global q_copy
        q_copy = q.copy()
        
        restart = False
        # get random state to start from 
        state = get_random_state(states)
        
        dealer_card, dealer_hand, player_hand = get_player_and_dealer_hands_from_state(state)
        
        while not restart:
            
            # choose random action from the dictionary
            # action_key = np.random.choice(actions.keys())
            action_key = get_gli_epolicy(q, state, 0.7)
            # action_key = exploration_function(q, state, nstate_action[state, ])
            action = actions[action_key]

            # if the action is hit then we add a random card to the players hand            
            if action_key == 'hit':
                player_hand = add_card_to_hand(player_hand, get_random_card())
                
                # check if player has busted
                if get_hand_value(player_hand) > 21:
                    # increment state-action pair in freq table
                    nstate_action[(state, action)] = nstate_action[(state, action)] + 1.0
                    # update q-table
                    q[state, action] = q[state, action] + discount_function(nstate_action[(state, action)]) \
                                       *((-1.0) + 0.0 - q[state, action])
                    restart = True
                    print('Busted')
                    break
                elif get_hand_value(player_hand) == 21:
                    # increment state-action pair in freq table
                    nstate_action[(state, action)] = nstate_action[(state, action)] + 1.0
                    # update q-table
                    q[state, action] = q[state, action] + discount_function(nstate_action[(state, action)]) \
                                       *((1.0) + 0.0 - q[state, action])
                    restart = True
                    print('Yay')
                    break
                else:
                    # increment state-action pair in freq table
                    nstate_action[(state, action)] = nstate_action[(state, action)] + 1.0
                    # find q_max for the next state
                    next_state = get_state_from_dealer_card_and_player_hand(dealer_card, player_hand)
                    q_max_val = q_max(q, next_state)
                    # update q-table
                    q[state, action] = q[state, action] + discount_function(nstate_action[(state, action)]) \
                                       *(0.0 + discount * q_max_val - q[state, action])
                    # update state
                    state = next_state
                    print('Next state')
                    
            # allow the dealer to play
            else:
                dealer_hand = deal_dealer_hand(dealer_hand)
                # calculate immediate reward
                reward = calculate_reward(player_hand, dealer_hand)
                # increment state-action pair in freq table
                nstate_action[(state, action)] = nstate_action[(state, action)] + 1.0
                # update q-table
                # q_max will be zero because this will be the terminal state
                q[state, action] = q[state, action] + discount_function(nstate_action[(state, action)]) \
                                       *(reward + 0.0 - q[state, action])
                restart = True
                break
                
        delta = np.max( np.abs( np.array(sorted(q.values()))-np.array(sorted(q_copy.values())) ) )
        print( 'iteration = ' + str(iteration) + ', ' + 'delta = ' + '%7.4f'%(delta) )
        iteration = iteration + 1
        
    return q

q = run_q_learning()
                
print_policy(q)