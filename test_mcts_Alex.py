# Adapté de https://medium.com/@_michelangelo_/monte-carlo-tree-search-mcts-alphazero-and-hopefully-muzero-for-dummies-11ad5d95d9d8

import math
import copy
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

GAME_NAME = 'ALE/Othello-v5'

env = gym.make(GAME_NAME)

GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]
print('In the ' + GAME_NAME + ' environment there are: ' + str(GAME_ACTIONS) + ' possible actions.')
# 2 actions possibles (right ou left)
print('In the ' + GAME_NAME + ' environment the observation is composed of: ' + str(GAME_OBS) + ' values.')
# observation = 4 valeurs (cart position, cart velocity, pole angle, pole angular velocity)

env.reset()
env.close()


class Node:
    '''
    The Node class represents a node of the MCTS tree.
    It contains the information needed for the algorithm to run its search.
    '''

    def __init__(self, game, done, parent, observation, action_index):
        # child nodes i.e. the next states of the game
        self.child = None

        # total rewards from MCTS exploration (the sum of the value of the rollouts that have been started from this node)
        self.T = 0

        # visit count
        self.N = 0

        # the environment in the current state represented by the node
        # Copy of the original game that we are currently playing, so that we can use it to simulate the search
        # current state of the game that will result in applying all the actions from the root node to the node itself
        self.game = game

        # observation of the environment
        # current state of the game in that node (with 4 values)
        self.observation = observation

        # if game is won/loss/draw
        self.done = done

        # link to parent node (for backpropagation)
        self.parent = parent

        # action index that leads to this node (i.e. is the action that the parent of the node has taken to get into this node)
        self.action_index = action_index

    def getUCBscore(self):

        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.
        '''
        # balances exploitation (i.e. pick the best known action) and exploration (i.e. explore new actions)

        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')

        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        # We use one of the possible MCTS formula for calculating the node value
        # first term = exploitation = current estimation value of the node
        # second term (without c=2) = exploration = inversely proportional to the number of times the node has been visited
        # c = tunable constant
        return (self.T / self.N) + 2 * math.sqrt(math.log(top_node.N) / self.N)

    def create_child(self):

        '''
        We create one children for each possible action of the game,
        then we apply such action to a copy of the current node enviroment
        and create such child node with proper information returned from the action executed
        '''

        if self.done:
            return

        actions = []
        games = []
        for i in range(GAME_ACTIONS):
            actions.append(i)
            new_game = copy.deepcopy(self.game)
            games.append(new_game)

        child = {}
        for action, game in zip(actions, games):
            observation, reward, done, _, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action)

        self.child = child

    def explore(self):

        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''

        # find a leaf node by choosing nodes with max U.

        current = self

        while current.child:

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length ", max_U)
            action = random.choice(actions)
            current = child[action]

        # play a random game, or expand if needed

        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.T = current.T + current.rollout()

        current.N += 1

        # update statistics and backpropagate

        parent = current

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):

        '''
        The rollout is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node.
        Taken alone, this value is quite random, but, the more rollouts we will do for such node,
        the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
        '''

        if self.done:
            return 0

        v = 0
        done = False
        new_game = copy.deepcopy(self.game)
        while not done:
            action = new_game.action_space.sample()
            observation, reward, done, _, _ = new_game.step(action)
            # observation, reward, terminated, truncated, info = env.step(action)
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break
        return v

    def next(self):
        # how to pick the next action after the search is done
        '''
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')

        child = self.child

        max_N = max(node.N for node in child.values())

        max_children = [c for a, c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length ", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index

    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None






MCTS_POLICY_EXPLORE = 100  # MCTS exploring constant: the higher, the more reliable, but slower in execution time

def Policy_Player_MCTS(mytree):

    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''

    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()

    next_tree, next_action = mytree.next()

    # note that here we are detaching the current node and returning the sub-tree
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!

    # j'ai commenté ça car y a pas de méthode detach_parent
    next_tree.detach_parent()

    return next_tree, next_action


episodes = 10
rewards = []
moving_average = []

'''
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our MCTS algorithm
- we will collect and plot the rewards to check if the MCTS is actually working.
'''

for e in range(episodes):

    reward_e = 0
    game = gym.make(GAME_NAME)
    observation = game.reset()
    done = False

    new_game = copy.deepcopy(game)
    mytree = Node(new_game, False, 0, observation, 0)

    print('episode #' + str(e + 1))

    while not done:

        mytree, action = Policy_Player_MCTS(mytree)

        observation, reward, done, _, _ = game.step(action)
        reward_e = reward_e + reward

        # game.render() # uncomment this if you want to see your agent in action!
        if done:
            print('reward_e ' + str(reward_e))
            game.close()
            break

    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print('moving average: ' + str(np.mean(rewards[-20:])))