import networkx as nx
from copy import deepcopy
from check import RESET_PENALTY, REFUEL_PENALTY, DROP_IN_DESTINATION_REWARD
from itertools import product

ids = ["322691718", "313539850"]


class Agent:
    """ This is a general Agent class that has all the common operation of the Agents. Helps avoid code duplication. """
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
        self.map = deepcopy(initial["map"])
        self.map_shape = (len(self.map), len(self.map[0]))
        del initial["map"]
        self.initial_state = deepcopy(initial)
        self.state = deepcopy(initial)
        self.graph = self.build_graph()
        self.taxi_names = [taxi_name for taxi_name in self.state["taxis"].keys()]
        self.passengers_names = list(self.state["passengers"].keys())

    def encode_state(self):
        """
        encode state in a safe format
        :return:
        """
        taxis = tuple((taxi_name, taxi_details["location"], taxi_details["fuel"], taxi_details["capacity"])
                      for taxi_name, taxi_details in self.state["taxis"].items())

        passengers = tuple((passenger_name, passenger_details["location"], passenger_details["destination"])
                           for passenger_name, passenger_details in self.state["passengers"].items())

        new_state = (taxis, passengers)

        return new_state

    def decode_state(self, state=None, inplace=True):
        """
        return state to operational format.
        :param state: the state | None
        :param inplace: if True - decode ~self.state~, else decode given state
        :return: if inplace = False - return Operational state, else return None
        """
        # TODO: decide with Orly how to represent states and encode / decode them, especially in this function!
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

    def transition(self, state, action, new_state):
        """ The function that calculates the probability of getting to new_state from state by performing action. """
        probability = 1
        self.decode_state(self.result(state, action))
        new_state = self.decode_state(new_state, inplace=False)
        for p in self.passengers_names:
            pc = self.state[p]["prob_change_goal"]
            probability *= pc * (1 / len(self.state[p]["possible_goals"])) + \
                           (self.state[p]["destination"] == new_state[p]["destination"]) * (1 - pc)

        return probability

    def is_action_legal(self, action, taxis_location_dict):
        """
        check if the action is legal to perform on ~self.state~
        """
        def _is_move_action_legal(move_action):
            taxi_name = move_action[1]
            if taxi_name not in self.state['taxis'].keys():
                return False
            if self.state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = self.state['taxis'][taxi_name]['location']
            l2 = move_action[2]
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
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['destination']:
                return False
            return True

        def _is_refuel_action_legal(refuel_action):
            """
            check if taxi in gas location
            """
            taxi_name = refuel_action[1]
            i, j = self.state['taxis'][taxi_name]['location']
            if self.map[i][j] == 'G':
                return True
            else:
                return False

        def _is_action_mutex(global_action):
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
        if len(action) != len(self.state["taxis"].keys()):
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

    def get_available_actions(self, taxi_name):
        """
        Return all possible moves that this taxi can make in ~self.state~.
        :param taxi_name: name of taxi.
        :return: list of moves for this taxi.
        """
        available_moves = []

        # "move" actions
        # TODO: Try to make a simpler check on 'move' action by using the graph.neighbors function.
        row = self.state["taxis"][taxi_name]["location"][0]
        column = self.state["taxis"][taxi_name]["location"][1]
        if self.state["taxis"][taxi_name]["fuel"] > 0:
            if row > 0 and self.map[row - 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row - 1, column)))
            if row < self.map_shape[0] - 1 and self.map[row + 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row + 1, column)))
            if column > 0 and self.map[row][column - 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column - 1)))
            if column < self.map_shape[1] - 1 and self.map[row][column + 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column + 1)))

        # "refuel" actions
        if self.map[row][column] == "G":
            available_moves.append(("refuel", taxi_name))

        # "pick up" actions
        if 0 < self.state["taxis"][taxi_name]["capacity"]:
            for passenger in self.passengers_names:
                if self.state["passengers"][passenger]["location"] == (row, column) and \
                        self.state["passengers"][passenger]["location"] != \
                        self.state["passengers"][passenger]["destination"]:
                    available_moves.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for passenger in self.passengers_names:
            if self.state["passengers"][passenger]["location"] == taxi_name and \
                    self.state["passengers"][passenger]["destination"] == (row, column):
                available_moves.append(("drop off", taxi_name, passenger))

        available_moves.append(("wait", taxi_name))
        return available_moves

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        if state is not None:
            self.decode_state(state)

        taxi_loc = dict((taxi_name, taxi_values["location"]) for taxi_name, taxi_values in self.state["taxis"].items())

        moves = []
        for taxi in self.taxi_names:
            moves.append(self.get_available_actions(taxi))

        for action in product(*moves):
            if self.is_action_legal(action, taxis_location_dict=taxi_loc):
                yield action
        yield "reset"
        yield "terminate"

    def result(self, state=None, action=None, return_encoded=True):
        """ Returns the state that results from executing the given action in the given state.
        The action must be one of self.actions(state). """
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

        if return_encoded:
            return self.encode_state()
        else:
            return

    def possible_results(self, state, action):
        """ Returns all possible states that result from executing the given action in the given state.
        The action must be one of self.actions(state). """
        base_result = self.result(state, action)
        results = [base_result, ]
        # TODO: complete this and try to optimize (use yield instead of returning the whole list)
        return results

    def act(self, state):
        """ The Agent's policy function. """
        raise NotImplemented

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


class OptimalTaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        self.states = self.get_all_states()

    def get_all_states(self, visited=None):
        # TODO: finish this basic search algorithm that finds all the states of the game
        if visited is None:
            visited = {self.encode_state(): 1}
        successors = set()
        for action in self.actions(self.state):
            child_states = self.possible_results(self.state, action)

    def act(self, state):
        raise NotImplemented


class TaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)

    def act(self, state):
        raise NotImplemented
