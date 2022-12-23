import search
import random
import math
from itertools import product

ids = ["322691718", "313539850"]


def is_legal(state, action):
    """
    check if an action is legal in a given state, so that no 2 taxis end up in the same area.
    :param state: current state
    :param action: action to check
    :return: True if legal, else False
    """
    taxis = dict((taxi_name, taxi_details["location"]) for taxi_name, taxi_details in state["taxis"].items())
    for taxi_action in action:
        if taxi_action[0] == "move":
            taxis[taxi_action[1]] = taxi_action[2]

    check_list = list(taxis.values())
    check_set = set()
    for location in check_list:
        if location in check_set:
            return False
        else:
            check_set.add(location)
    return True


NOT_PICKED_UP = 0
PICKED_UP = 1
DELIVERED = 2


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map_shape = (len(initial["map"]), len(initial["map"][0]))
        self.n_taxis = len(initial["taxis"].keys())
        for taxi_name in initial["taxis"].keys():
            initial["taxis"][taxi_name]["traveling"] = []
            initial["taxis"][taxi_name]["fuel_capacity"] = initial["taxis"][taxi_name]["fuel"]
        for passenger in initial["passengers"].keys():
            initial["passengers"][passenger]["status"] = NOT_PICKED_UP
        self.state = initial
        initial = self.hash_state()
        search.Problem.__init__(self, initial)

    def hash_state(self):
        taxis = tuple((taxi_name, taxi_details["location"], taxi_details["fuel"], frozenset(taxi_details["traveling"]))
                      for taxi_name, taxi_details in self.state["taxis"].items())

        passengers = tuple((passenger_name, passenger_details["location"], passenger_details["status"])
                           for passenger_name, passenger_details in self.state["passengers"].items())

        new_state = (taxis, passengers)

        return new_state

    def unhash_state(self, state):
        for taxi_name, location, fuel, traveling in state[0]:
            self.state["taxis"][taxi_name]["location"] = location
            self.state["taxis"][taxi_name]["fuel"] = fuel
            self.state["taxis"][taxi_name]["traveling"] = list(traveling)

        for passenger_name, location, status in state[1]:
            self.state["passengers"][passenger_name]["location"] = location
            self.state["passengers"][passenger_name]["status"] = status

    def get_available_moves(self, taxi_name, state):
        """
        Return all possible moves that this taxi can make.
        :param taxi_name: name of taxi.
        :param state: current state, (unhashed).
        :return: list of moves for this taxi.
        """
        available_moves = []

        # "move" actions
        row = state["taxis"][taxi_name]["location"][0]
        column = state["taxis"][taxi_name]["location"][1]
        if state["taxis"][taxi_name]["fuel"] > 0:
            if row > 0 and state["map"][row - 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row - 1, column)))
            if row < self.map_shape[0] - 1 and state["map"][row + 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row + 1, column)))
            if column > 0 and state["map"][row][column - 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column - 1)))
            if column < self.map_shape[1] - 1 and state["map"][row][column + 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column + 1)))

        # "refuel" actions
        if state["map"][row][column] == "G":
            available_moves.append(("refuel", taxi_name))

        # "pick up" actions
        if len(state["taxis"][taxi_name]["traveling"]) < state["taxis"][taxi_name]["capacity"]:
            for passenger in state["passengers"].keys():
                if state["passengers"][passenger]["location"][0] == row and \
                        state["passengers"][passenger]["location"][1] == column:
                    available_moves.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for traveller in state["taxis"][taxi_name]["traveling"]:
            if state["passengers"][traveller]["destination"][0] == row and \
                    state["passengers"][traveller]["destination"][1] == column:
                available_moves.append(("drop off", taxi_name, traveller))

        available_moves.append(("wait", taxi_name))
        return available_moves

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        state = unhash_state(state)

        taxis = list(state["taxis"].keys())
        moves = []
        for taxi in taxis:
            moves.append(self.get_available_moves(taxi, state))

        for action in product(*moves):
            if is_legal(state, action):
                yield action

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        next_state = unhash_state(state)
        for taxi_action in action:
            if taxi_action[0] == "move":
                next_state["taxis"][taxi_action[1]]["location"] = taxi_action[2]
                next_state["taxis"][taxi_action[1]]["fuel"] -= 1
                for traveller in next_state["taxis"][taxi_action[1]]["traveling"]:
                    next_state["passengers"][traveller]["location"] = taxi_action[2]
            elif taxi_action[0] == "pick up":
                next_state["taxis"][taxi_action[1]]["traveling"].append(taxi_action[2])
                next_state["passengers"][taxi_action[2]]["status"] = PICKED_UP
            elif taxi_action[0] == "drop off":
                next_state["taxis"][taxi_action[1]]["traveling"].remove(taxi_action[2])
                next_state["passengers"][taxi_action[2]]["status"] = DELIVERED
            elif taxi_action[0] == "refuel":
                next_state["taxis"][taxi_action[1]]["fuel"] = next_state["taxis"][taxi_action[1]]["fuel_capacity"]
        return hash_state(next_state)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state = unhash_state(state)
        for passenger in state["passengers"].keys():
            if state["passengers"][passenger]["status"] != DELIVERED:
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return self.h_1(node)

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = unhash_state(node.state)
        counter = 0
        for passenger in state["passengers"].keys():
            counter += 2 - state["passengers"][passenger]["status"]
        return counter / self.n_taxis

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = unhash_state(node.state)
        counter = 0
        for passenger in state["passengers"].keys():
            counter += abs(state["passengers"][passenger]["location"][0] -
                           state["passengers"][passenger]["destination"][0]) + \
                       abs(state["passengers"][passenger]["location"][1] -
                           state["passengers"][passenger]["destination"][1])
        return counter / self.n_taxis

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_taxi_problem(game):
    return TaxiProblem(game)
