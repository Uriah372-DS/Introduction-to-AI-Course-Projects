import random

import networkx as nx
from copy import deepcopy
from itertools import product

ids = ["322691718", "313539850"]

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100


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
        self.state = deepcopy(self.initial_state)
        self.graph = self.build_graph()
        self.taxi_names = list(self.state["taxis"].keys())
        self.passengers_names = list(self.state["passengers"].keys())
        self.encoded_initial_state = self.encode_state(self.initial_state)

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
        encode state in a safe format.
        :returns: tuple.
            the encoded state.
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
            for taxi_name in self.taxi_names)

        passengers = tuple(
            (
                passenger_name,
                self.state["passengers"][passenger_name]["location"],
                self.state["passengers"][passenger_name]["destination"]
            )
            for passenger_name in self.passengers_names)

        new_state = (taxis, passengers, self.state["turns to go"])

        return new_state

    def decode_state(self, state=None, into_self_state=True):
        """
        return state to operational format.
        :param state: the state | None
        :param into_self_state: if True - decode state contents into ~self.state~,
            else decode state contents into a copy of ~self.state~.
        :return: if into_self_state = False - return Operational state, else return None
        """
        if into_self_state:
            for taxi_name, location, fuel, capacity in state[0]:
                self.state["taxis"][taxi_name]["location"] = location
                self.state["taxis"][taxi_name]["fuel"] = fuel
                self.state["taxis"][taxi_name]["capacity"] = capacity

            for passenger_name, location, destination in state[1]:
                self.state["passengers"][passenger_name]["location"] = location
                self.state["passengers"][passenger_name]["destination"] = destination
            self.state["turns to go"] = state[2]
            return self.state
        else:
            new_state = deepcopy(self.state)
            for taxi_name, location, fuel, capacity in state[0]:
                new_state["taxis"][taxi_name]["location"] = location
                new_state["taxis"][taxi_name]["fuel"] = fuel
                new_state["taxis"][taxi_name]["capacity"] = capacity

            for passenger_name, location, destination in state[1]:
                new_state["passengers"][passenger_name]["location"] = location
                new_state["passengers"][passenger_name]["destination"] = destination
            new_state["turns to go"] = state[2]
            return new_state

    def reward(self, action):
        """ The function that calculates reward of performing this action on a state. """
        if action == "reset":
            return -RESET_PENALTY
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
        """
        The function that calculates the probability of getting to new_state from state by performing action.
        :param state: current state, in dict format.
        :param action: the action to perform.
        :param new_state: possible new state, in dict format.
        :return: float.
            the probability to reach new_state from state using action.
        """

        probability = 1
        if action == "terminate":
            probability = float(new_state["turns to go"] == 0)
        elif action == "reset":
            self.result(state, action, into_self_state=True, return_encoded=False)
            encoded_new_state = self.encode_state(new_state)
            probability = float(self.encoded_initial_state[0] == encoded_new_state[0] and
                                self.encoded_initial_state[1] == encoded_new_state[1])
        else:
            self.result(state, action, into_self_state=True, return_encoded=False)

            for passenger in self.passengers_names:
                pc = self.state["passengers"][passenger]["prob_change_goal"]

                if self.state["passengers"][passenger]["destination"] in \
                        self.state["passengers"][passenger]["possible_goals"]:
                    probability *= (pc / len(self.state["passengers"][passenger]["possible_goals"])) + (
                            self.state["passengers"][passenger]["destination"] ==
                            new_state["passengers"][passenger]["destination"]) * (1 - pc)
                else:
                    if self.state["passengers"][passenger]["destination"] == \
                            new_state["passengers"][passenger]["destination"]:
                        probability *= 1 - pc
                    else:
                        probability *= (pc / len(self.state["passengers"][passenger]["possible_goals"]))

        return probability

    def is_action_legal(self, state, action, taxis_location_dict):
        """
        check if the action is legal to perform on ~self.state~.
        """

        def _is_move_action_legal(atomic_move_action):
            taxi_name = atomic_move_action[1]
            if taxi_name not in self.taxi_names:
                return False
            if state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = atomic_move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            # check passenger is not in his destination
            if state['passengers'][passenger_name]['destination'] == \
                    state['passengers'][passenger_name]['location']:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            """ check same position. """
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            if state['taxis'][taxi_name]['location'] == state['passengers'][passenger_name]['destination']:
                return True
            return False

        def _is_refuel_action_legal(refuel_action):
            """ check if taxi in gas location. """
            taxi_name = refuel_action[1]
            i, j = state['taxis'][taxi_name]['location']
            if self.map[i][j] == 'game_tree':
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
        if len(state['taxis']) > 1:

            move_actions = [a for a in action if a[0] == 'move']

            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]

            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                return False
        return True

    def get_atomic_actions(self, state, taxi_name):
        """
        Return all possible actions that the given taxi can make in ~self.state~.

        :param state: the current state, in dict format.
        :param taxi_name: name of the taxi.
        :return: list of moves for the taxi.
        """
        available_actions = []

        # "move" actions
        # TODO: Try to make a simpler check on 'move' action by using the graph.neighbors function.
        row = state["taxis"][taxi_name]["location"][0]
        column = state["taxis"][taxi_name]["location"][1]
        if state["taxis"][taxi_name]["fuel"] > 0:
            if row > 0 and self.map[row - 1][column] != 'I':
                available_actions.append(("move", taxi_name, (row - 1, column)))
            if row < self.map_shape[0] - 1 and self.map[row + 1][column] != 'I':
                available_actions.append(("move", taxi_name, (row + 1, column)))
            if column > 0 and self.map[row][column - 1] != 'I':
                available_actions.append(("move", taxi_name, (row, column - 1)))
            if column < self.map_shape[1] - 1 and self.map[row][column + 1] != 'I':
                available_actions.append(("move", taxi_name, (row, column + 1)))

        # "refuel" actions
        if self.map[row][column] == "game_tree":
            available_actions.append(("refuel", taxi_name))

        # "pick up" actions
        if 0 < state["taxis"][taxi_name]["capacity"]:
            for passenger in self.passengers_names:
                if state["passengers"][passenger]["location"] == (row, column) and \
                        state["passengers"][passenger]["location"] != \
                        state["passengers"][passenger]["destination"]:
                    available_actions.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for passenger in self.passengers_names:
            if state["passengers"][passenger]["location"] == taxi_name and \
                    state["passengers"][passenger]["destination"] == (row, column):
                available_actions.append(("drop off", taxi_name, passenger))

        available_actions.append(("wait", taxi_name))
        return available_actions

    def actions(self, state):
        """
        Returns all the actions that can be executed in the given state.
        The result should be a tuple (or other iterable) of actions
        as defined in the problem description file.

        :param state: the current state, in dict format.
        """

        taxi_loc = dict((taxi_name, taxi_values["location"]) for taxi_name, taxi_values in state["taxis"].items())
        for action in filter(lambda a: self.is_action_legal(state, a, taxis_location_dict=taxi_loc),
                             product(*[self.get_atomic_actions(state, taxi) for taxi in self.taxi_names])):
            yield action

        yield "reset"
        yield "terminate"

    def result(self, state, action, into_self_state=True, return_encoded=True):
        """
        Returns the deterministic state that results from executing the given action in the given state.
        The action must be one of self.actions(state).

        :param state: the current state, in dict format.
        :param action: the action to perform.
        :param into_self_state: if True override ~self.state~ with resulting state,
            else return a deepcopy of ~self.state~ with resulting state changes.
        :param return_encoded: if True return encoded result, else return in dict format.
        """
        state_copy = deepcopy(state)
        if into_self_state:
            self.state = state_copy
            new_state = self.state
        else:
            new_state = state_copy

        ttg = new_state["turns to go"]
        if action == "reset":
            new_state = deepcopy(self.initial_state)
            new_state["turns to go"] = ttg - 1
        elif action == "terminate":
            new_state = deepcopy(self.initial_state)
            new_state["turns to go"] = 0
        else:
            for taxi_action in action:
                if taxi_action[0] == "move":
                    new_state["taxis"][taxi_action[1]]["location"] = taxi_action[2]
                    new_state["taxis"][taxi_action[1]]["fuel"] -= 1
                elif taxi_action[0] == "pick up":
                    new_state["passengers"][taxi_action[2]]["location"] = taxi_action[1]
                    new_state["taxis"][taxi_action[1]]["capacity"] -= 1
                elif taxi_action[0] == "drop off":
                    new_state["passengers"][taxi_action[2]]["location"] = new_state["taxis"][taxi_action[1]]["location"]
                    new_state["taxis"][taxi_action[1]]["capacity"] += 1
                elif taxi_action[0] == "refuel":
                    new_state["taxis"][taxi_action[1]]["fuel"] = self.initial_state["taxis"][taxi_action[1]]["fuel"]

            new_state["turns to go"] = ttg - 1

        if return_encoded:
            return self.encode_state(new_state)
        else:
            return new_state

    def possible_results(self, state: dict, action):
        """
        Returns all possible game_tree that result from executing the given action in the given state.
        The action must be one of self.actions(state).
        :param state: the current state, in dict format.
        :param action: the action to perform.
        """
        if action == "terminate" or action == "reset":
            terminal_state = self.result(state=state, action=action, into_self_state=False, return_encoded=False)
            return [terminal_state, ]

        state = self.result(state, action, into_self_state=False, return_encoded=False)
        possible_destinations = [list(set(state["passengers"][p]["possible_goals"]).union(
            {state["passengers"][p]["destination"]})) for p in self.passengers_names]
        for destinations in product(*possible_destinations):

            for passenger, destination in zip(self.passengers_names, destinations):
                state["passengers"][passenger]["destination"] = destination

            yield deepcopy(state)

    def act(self, state):
        """ The Agent's policy function. """
        raise NotImplemented


class OptimalTaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        self.game_tree = nx.DiGraph()
        self.build_game_tree(state=self.initial_state,
                             policy_finder=self.value_iteration)
        print("initial state value: ", self.game_tree.nodes[self.encode_state(self.initial_state)]['value'])

    def build_game_tree(self, state, policy_finder):
        """
        Recursively build a game tree represented as a NetworkX DiGraph, starting from the given initial state.
        The function uses a depth-first search algorithm to explore the tree.

        Parameters
        ----------
        state : dict
            The state of the game to explore.
        policy_finder : function
            the method that finds the optimal policy.

        Returns
        -------
            None
        """

        # rewind on terminal states
        s = self.encode_state(state)
        if state["turns to go"] == 0:
            self.game_tree.add_node(s,
                                    value=0.0,
                                    policy=None)
            return

        s = self.encode_state(state)
        # add current state to the game tree
        self.game_tree.add_node(s, value=0.0, policy=None)
        # for every performable action, explore all possible successors of the current state
        for action in self.actions(state):
            # calculate the reward for the current action
            action_reward = self.reward(action)
            # for every unexplored child state, add it's game subtree to the game tree and then connect it to this state
            for sweet_summer_child in self.possible_results(state, action):
                s_child = self.encode_state(sweet_summer_child)
                # explore child subtree if not explored yet
                if s_child not in self.game_tree.nodes:
                    self.build_game_tree(state=sweet_summer_child,
                                         policy_finder=policy_finder)
                # connect subtree to main game tree
                self.game_tree.add_edge(u_of_edge=s,
                                        v_of_edge=s_child,
                                        action=action,
                                        reward=action_reward,
                                        transition_probability=self.transition(state,
                                                                               action,
                                                                               sweet_summer_child))
        policy_finder(s)

    def value_iteration(self, s):
        # Compute the value of node s as the maximum expected value over all outgoing edges (s, u) from s
        successors = list(self.game_tree.successors(s))
        if successors:
            self.game_tree.nodes[s]['value'], self.game_tree.nodes[s]['policy'] = max(
                (self.game_tree[s][u]['reward'] + self.game_tree.nodes[u]['value'],
                 self.game_tree.edges[s, u]['action'])
                for u in successors)
        else:
            self.game_tree.nodes[s]['policy'] = max(list(self.actions(self.decode_state(s))),
                                                    key=lambda a: self.reward(a))
            self.game_tree.nodes[s]['value'] = self.reward(self.game_tree.nodes[s]['policy'])

    def policy_iteration(self, discount_factor=1.0):
        """
        Perform policy iteration on a game tree represented as a NetworkX DiGraph.
        The actions between game_tree are given as edge attributes on the DiGraph.
        The values and optimal policy are saved as node attributes on the DiGraph.

        Parameters
        ----------
        discount_factor : float (default: 1.0)
            The discount factor to use in the policy iteration algorithm.

        Returns
        -------
        None
        """
        # Initialize the values of all nodes to 0
        for node in self.game_tree.nodes():
            self.game_tree.nodes[node]["value"] = 0.0

        # Initialize the policy of all nodes to None
        for node in self.game_tree.nodes():
            self.game_tree.nodes[node]["policy"] = None

        # Set the values of the leaf nodes (terminal game_tree)
        for node in self.game_tree.nodes():
            if not self.game_tree.successors(node):
                # Set the value of the node to the reward associated with the edge leading to the node
                self.game_tree.nodes[node]["value"] = \
                    self.game_tree[node][self.game_tree.predecessors(node)[0]]["reward"]

        # Perform policy iteration
        while True:
            # Evaluate the current policy
            stable = True
            for node in nx.dfs_postorder_nodes(self.game_tree):
                if not list(self.game_tree.successors(node)):
                    # Skip leaf nodes
                    continue

                # Calculate the expected value of each action based on the values of the resulting game_tree
                action_values = {}
                for successor, edge_attributes in self.game_tree[node].items():
                    action_values[edge_attributes['action']] = self.game_tree.nodes[successor]["value"] * \
                                                               edge_attributes["transition_probability"]

                # Select the action with the highest expected value as the optimal action for the current state
                optimal_action = max(action_values, key=action_values.get)

                # Update the value and policy of the node if they have changed
                if self.game_tree.nodes[node]["policy"] != optimal_action:
                    self.game_tree.nodes[node]["policy"] = optimal_action
                    stable = False
                if abs(self.game_tree.nodes[node]["value"] - action_values[optimal_action]) > 1e-6:
                    self.game_tree.nodes[node]["value"] = action_values[optimal_action]
                    stable = False

            if stable:
                # The policy has converged, so we can stop iterating
                break

        # Discount the values of the nodes based on the discount factor
        for node in self.game_tree.nodes():
            if self.game_tree.successors(node):
                self.game_tree.nodes[node]["value"] /= discount_factor

    def hybrid_iteration(self, state):
        pass

    def act(self, state):
        return self.game_tree.nodes[self.encode_state(state)]["policy"]


class TaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        self.default_agent = OptimalTaxiAgent(initial)

    def act(self, state):
        return self.default_agent.act(state)
