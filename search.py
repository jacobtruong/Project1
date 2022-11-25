# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Dictionary was attempted to be used first, but it complicates the process of checking for different problems types
    # Create an empty list of visited node
    visited = []

    # Get first node
    node = problem.getStartState()

    # Create a stack
    stack = util.Stack()

    # Push into the stack the initial start node and empty action list
    stack.push((node, []))

    # While stack is not empty, continue traversing
    while not stack.isEmpty():
        n, a = stack.pop()

        # If node has not been visited
        if n not in visited:
            # Add node to the visited node list
            visited.append(n)

            # If the current node is the goal state, return the steps (actions) to get here
            if problem.isGoalState(n):
                return a

            # Find each neighbour of n and add them into the stack along with the action list to get there
            for neighbour in problem.getSuccessors(n):

                # The neighbour is a tuple of 3: the next node (a tuple of x and y), the direction, and the cost.
                # ie: neighbour == ((5,4), 'West', 1)). In this case, cost is unused
                if neighbour[0] not in visited:
                    stack.push((neighbour[0], a + [neighbour[1]]))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Dictionary was attempted to be used first, but it complicates the process of checking for different problems types
    # Create an empty list of visited node
    visited = []

    # Get first node
    node = problem.getStartState()

    # Create a queue
    queue = util.Queue()

    # Push into the queue the initial start node and empty action list
    queue.push((node, []))


    # While queue is not empty, continue traversing
    while not queue.isEmpty():
        n, a = queue.pop()

        # If node has not been visited
        if n not in visited:
            # Add node to the visited node list
            visited.append(n)

            # If the current node is the goal state, return the steps (actions) to get here
            if problem.isGoalState(n):
                return a

            # Find each neighbour of n and add them into the stack along with the action list to get there
            for neighbour in problem.getSuccessors(n):

                # The neighbour is a tuple of 3: the next node (a tuple of x and y), the direction, and the cost.
                # ie: neighbour == ((5,4), 'West', 1)). In this case, cost is unused
                    queue.push((neighbour[0], a + [neighbour[1]]))


    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Dictionary was attempted to be used first, but it complicates the process of checking for different problems types
    # Create an empty list of visited node
    visited = []

    # Get first node
    node = problem.getStartState()

    # Create a priorityQueue
    priorityQueue = util.PriorityQueue()

    # Push into the queue the initial start node, empty action list, and cost
    priorityQueue.push((node, [], 0), 0)

    # While queue is not empty, continue traversing
    while not priorityQueue.isEmpty():
        n, a, c = priorityQueue.pop()

        # If node has not been visited
        if n not in visited:
            # Add node to the visited node list
            visited.append(n)

            # If the current node is the goal state, return the steps (actions) to get here
            if problem.isGoalState(n):
                return a

            # Find each neighbour of n and add them into the stack along with the action list to get there
            for neighbour in problem.getSuccessors(n):
                # The neighbour is a tuple of 3: the next node (a tuple of x and y), the direction, and the cost.
                # ie: neighbour == ((5,4), 'West', 1)). In this case, cost is unused
                priorityQueue.push((neighbour[0], a + [neighbour[1]], c + neighbour[2]), c + neighbour[2])

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Dictionary was attempted to be used first, but it complicates the process of checking for different problems types
    # Create an empty list of visited node
    visited = []

    # Get first node
    node = problem.getStartState()

    # Create a priorityQueue
    priorityQueue = util.PriorityQueue()

    # Push into the queue the initial start node, empty action list, and cost
    # priorityQueue.push((node, [], 0), heuristic(node, problem))
    priorityQueue.push((node, [], 0), 0)

    # While queue is not empty, continue traversing
    while not priorityQueue.isEmpty():
        n, a, c = priorityQueue.pop()

        # If node has not been visited
        if n not in visited:
            # Add node to the visited node list
            visited.append(n)

            # If the current node is the goal state, return the steps (actions) to get here
            if problem.isGoalState(n):
                return a

            # Find each neighbour of n and add them into the stack along with the action list to get there
            for neighbour in problem.getSuccessors(n):
                # The neighbour is a tuple of 3: the next node (a tuple of x and y), the direction, and the cost.
                # ie: neighbour == ((5,4), 'West', 1)). In this case, cost is unused
                priorityQueue.push((neighbour[0], a + [neighbour[1]], c + neighbour[2]), c + neighbour[2] + heuristic(neighbour[0], problem))

    util.raiseNotDefined()


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
