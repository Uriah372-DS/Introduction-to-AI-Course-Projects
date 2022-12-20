import networkx as nx
import numpy as np
from copy import deepcopy
from check import RESET_PENALTY, REFUEL_PENALTY, DROP_IN_DESTINATION_REWARD
from itertools import product

ids = ["322691718", "313539850"]


class Agent:
    """ This is a general Agent class that has all the common operations of the Agents. Avoids code duplication. """

    def __init__(self, initial):
        """
        Base constructor.
        Creates attributes:
        - initial_state: a copy of initial
        - state: the current state that is being operated on.
        - map_shape: the dimensions of the game map.
        - graph: a graph representation of the game map. useful of calculating distance that includes obstacles.
        - taxi_names: a list of all the taxis, for iterating on.
        - passengers_names: a list of all the passengers, for iterating on.
        :param initial: Starting game state.
        """
        initial_copy = deepcopy(initial)
        self.map = deepcopy(initial_copy["map"])
        self.map_shape = (len(self.map), len(self.map[0]))
        del initial_copy["map"]
        self.initial_state = deepcopy(initial_copy)
        self.transitions()
        self.state = deepcopy(self.initial_state)
        self.graph = self.build_graph()
        self.taxi_names = list(self.state["taxis"].keys())
        self.passengers_names = list(self.state["passengers"].keys())

    def build_graph(self):
        """ build the graph of the problem. """
        g = nx.grid_graph(self.map_shape)
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def encode_state(self, state=None):
        """
        encode state in a safe format
        :returns: tuple.
            the encoded state
        """
        if state:
            self.state = state
        taxis = tuple(
            (
                taxi_name,
                self.state["taxis"][taxi_name]["location"],
                self.state["taxis"][taxi_name]["fuel"],
                self.state["taxis"][taxi_name]["capacity"]
            )
            for taxi_name, taxi_details in self.taxi_names)

        passengers = tuple(
            (
                passenger_name,
                self.state["passengers"][passenger_name]["location"],
                self.state["passengers"][passenger_name]["destination"]
            )
            for passenger_name, passenger_details in self.passengers_names)

        new_state = (taxis, passengers)

        return new_state

    def decode_state(self, state=None, inplace=True):
        """
        return state to operational format.
        :param state: the state | None
        :param inplace: if True - decode ~self.state~, else decode given state
        :return: if inplace = False - return Operational state, else return None
        """
        # TODO: decide with Orly how to represent states and encode / decode them, decoding!
        if inplace:
            for taxi_name, location, fuel, capacity in state[0]:
                self.state["taxis"][taxi_name]["location"] = location
                self.state["taxis"][taxi_name]["fuel"] = fuel
                self.state["taxis"][taxi_name]["capacity"] = capacity

            for passenger_name, location, destination in state[1]:
                self.state["passengers"][passenger_name]["location"] = location
                self.state["passengers"][passenger_name]["destination"] = destination
        else:
            new_state = deepcopy(self.state)
            for taxi_name, location, fuel, capacity in state[0]:
                new_state["taxis"][taxi_name]["location"] = location
                new_state["taxis"][taxi_name]["fuel"] = fuel
                new_state["taxis"][taxi_name]["capacity"] = capacity

            for passenger_name, location, destination in state[1]:
                new_state["passengers"][passenger_name]["location"] = location
                new_state["passengers"][passenger_name]["destination"] = destination
            return new_state

    def reward(self, action):
        """ The function that calculates reward of performing this action on a state. """
        if action == "reset":
            return RESET_PENALTY
        elif action == "terminate":
            return 0
        score = 0
        for atomic_action in action:
            if atomic_action[0] == "drop off":
                score += DROP_IN_DESTINATION_REWARD
            elif atomic_action[0] == "refuel":
                score -= REFUEL_PENALTY
        return score

    def transitions(self):
        """ The function that calculates the probability of getting to new_state from state by performing action. """
        # # TODO: remove this function!
        # probability = 1
        # self.decode_state(self.result(state, action))
        # new_state = self.decode_state(new_state, inplace=False)
        # for p in self.passengers_names:
        #     pc = self.initial_state[p]["prob_change_goal"]
        #     probability *= pc * (1 / len(self.state[p]["possible_goals"])) + (
        #                 self.state[p]["destination"] == new_state[p]["destination"]) * (1 - pc)
        #
        # return probability

        for p in self.passengers_names:
            pc = self.initial_state[p]["prob_change_goal"]
            self.initial_state["passengers"][p]["prob_different_specific_goal"] = pc / len(self.state[p]["possible_goals"])
            self.initial_state["passengers"][p]["prob_same_goal"] = (pc / len(self.state[p]["possible_goals"])) + (1 - pc)

    def is_action_legal(self, action, taxis_location_dict):
        """
        check if the action is legal to perform on ~self.state~
        """

        def _is_move_action_legal(atomic_move_action):
            taxi_name = atomic_move_action[1]
            if taxi_name not in self.taxi_names:
                return False
            if self.state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = self.state['taxis'][taxi_name]['location']
            l2 = atomic_move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if self.state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            # check passenger is not in his destination
            if self.state['passengers'][passenger_name]['destination'] == \
                    self.state['passengers'][passenger_name]['location']:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            """ check same position. """
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            if self.state['taxis'][taxi_name]['location'] == self.state['passengers'][passenger_name]['destination']:
                return True
            return False

        def _is_refuel_action_legal(refuel_action):
            """ check if taxi in gas location. """
            taxi_name = refuel_action[1]
            i, j = self.state['taxis'][taxi_name]['location']
            if self.map[i][j] == 'G':
                return True
            else:
                return False

        def _is_action_mutex(global_action):
            # TODO: check if this condition is always false
            # one action per taxi
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            # pick up the same person
            pick_actions = [a for a in global_action if a[0] == 'pick up']
            if len(pick_actions) > 1:
                passengers_to_pick = set([a[2] for a in pick_actions])
                if len(passengers_to_pick) != len(pick_actions):
                    return True
            return False

        if action == "reset":
            return True
        if action == "terminate":
            return True

        if len(action) != len(self.taxi_names):
            return False

        for atomic_action in action:

            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    return False

            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action):
                    return False

            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action):
                    return False

            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action):
                    return False

            elif atomic_action[0] != 'wait':
                return False

        # check mutex action
        if _is_action_mutex(action):
            return False

        # check taxis collision
        if len(self.state['taxis']) > 1:

            move_actions = [a for a in action if a[0] == 'move']

            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]

            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                return False
        return True

    def get_atomic_actions(self, taxi_name):
        """
        Return all possible actions that the given taxi can make in ~self.state~.
        :param taxi_name: name of the taxi.
        :return: list of moves for the taxi.
        """
        available_actions = []

        # "move" actions
        # TODO: Try to make a simpler check on 'move' action by using the graph.neighbors function.
        row = self.state["taxis"][taxi_name]["location"][0]
        column = self.state["taxis"][taxi_name]["location"][1]
        if self.state["taxis"][taxi_name]["fuel"] > 0:
            if row > 0 and self.map[row - 1][column] != 'I':
                available_actions.append(("move", taxi_name, (row - 1, column)))
            if row < self.map_shape[0] - 1 and self.map[row + 1][column] != 'I':
                available_actions.append(("move", taxi_name, (row + 1, column)))
            if column > 0 and self.map[row][column - 1] != 'I':
                available_actions.append(("move", taxi_name, (row, column - 1)))
            if column < self.map_shape[1] - 1 and self.map[row][column + 1] != 'I':
                available_actions.append(("move", taxi_name, (row, column + 1)))

        # "refuel" actions
        if self.map[row][column] == "G":
            available_actions.append(("refuel", taxi_name))

        # "pick up" actions
        if 0 < self.state["taxis"][taxi_name]["capacity"]:
            for passenger in self.passengers_names:
                if self.state["passengers"][passenger]["location"] == (row, column) and \
                        self.state["passengers"][passenger]["location"] != \
                        self.state["passengers"][passenger]["destination"]:
                    available_actions.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for passenger in self.passengers_names:
            if self.state["passengers"][passenger]["location"] == taxi_name and \
                    self.state["passengers"][passenger]["destination"] == (row, column):
                available_actions.append(("drop off", taxi_name, passenger))

        available_actions.append(("wait", taxi_name))
        return available_actions

    def actions(self, state):
        """
        Returns all the actions that can be executed in the given state.
        The result should be a tuple (or other iterable) of actions
        as defined in the problem description file.
        """
        if state is not None:
            self.decode_state(state)

        taxi_loc = dict((taxi_name, taxi_values["location"]) for taxi_name, taxi_values in self.state["taxis"].items())

        yield filter(lambda a: self.is_action_legal(a, taxis_location_dict=taxi_loc),
                     product(*[self.get_atomic_actions(taxi) for taxi in self.taxi_names]))
        yield "reset"
        yield "terminate"

    def result(self, state=None, action=None, inplace=True, return_encoded=True):
        """ Returns the state that results from executing the given action in the given state.
        The action must be one of self.actions(state). """
        # TODO: can we replace this with the result function in check.py???
        if action == "reset":
            ttg = self.state["turns to go"]
            self.state = deepcopy(self.initial_state)
            self.state["turns to go"] = ttg
            return
        if action == "terminate":
            return
        if state is None:
            self.decode_state()
        for taxi_action in action:
            if taxi_action[0] == "move":
                self.state["taxis"][taxi_action[1]]["location"] = taxi_action[2]
                self.state["taxis"][taxi_action[1]]["fuel"] -= 1
            elif taxi_action[0] == "pick up":
                self.state["passengers"][taxi_action[2]]["location"] = taxi_action[1]
                self.state["taxis"][taxi_action[1]]["capacity"] -= 1
            elif taxi_action[0] == "drop off":
                self.state["passengers"][taxi_action[2]]["location"] = self.state["taxis"][taxi_action[1]]["location"]
                self.state["taxis"][taxi_action[1]]["capacity"] += 1
            elif taxi_action[0] == "refuel":
                self.state["taxis"][taxi_action[1]]["fuel"] = self.initial_state["taxis"][taxi_action[1]]["fuel"]

        self.state["turns to go"] -= 1

        if return_encoded:
            return self.encode_state()
        else:
            if not inplace:
                return deepcopy(self.state)

    def possible_results(self, state, action):
        """
        Returns all possible states that result from executing the given action in the given state.
        The action must be one of self.actions(state).
        Note: the state is given at dictionary format.
        """

        self.result(state, action, inplace=True, return_encoded=False)
        for destinations in product(*[set(self.state["passengers"][p]["possible_goals"]).union(
                self.state["passengers"][p]["destination"]) for p in self.passengers_names]):

            for passenger, destination in zip(self.passengers_names, destinations):
                self.state["passengers"][passenger]["destination"] = destination

            yield deepcopy(self.state)


    def act(self, state):
        """ The Agent's policy function. """
        raise NotImplemented


class OptimalTaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        self.states = nx.DiGraph()
        self.get_all_states(state=self.initial_state, visited={self.encode_state(self.initial_state)})

    def get_all_states(self, state, visited):
        if state["turns to go"] < 0:
            return
        s = self.encode_state(state)
        for action in self.actions(state):
            for sweet_summer_child in self.possible_results(state, action):
                s_child = self.encode_state(sweet_summer_child)
                self.states.add_edge(u_of_edge=s,
                                     v_of_edge=s_child,
                                     action=action,
                                     value=self.reward(action))
                if s_child not in visited:
                    visited.add(s_child)
                    self.get_all_states(state=sweet_summer_child, visited=visited)

    def value_iteration(self):
        pass

    def policy_iteration(self):
        pass

    def hybrid_iteration(self):
        pass

    def act(self, state):
        raise NotImplemented


class TaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)

    def act(self, state):
        raise NotImplemented
