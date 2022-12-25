import random

import networkx as nx
from copy import deepcopy
from itertools import product
from collections import deque

# import matplotlib.pyplot as plt; nx.draw(self.game_graph); plt.show()

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
        self.map_graph = self.build_map_graph()
        del initial_copy["map"]
        del initial_copy["optimal"]
        self.taxi_names = list(initial_copy["taxis"].keys())
        self.passengers_names = list(initial_copy["passengers"].keys())
        self.possible_goals = {p_name: initial_copy["passengers"][p_name]["possible_goals"]
                               for p_name in self.passengers_names}
        self.prob_change_goal = {p_name: initial_copy["passengers"][p_name]["prob_change_goal"]
                                 for p_name in self.passengers_names}
        for p_name in self.passengers_names:
            del initial_copy["passengers"][p_name]["possible_goals"]
            del initial_copy["passengers"][p_name]["prob_change_goal"]

        self.initial_state = initial_copy
        self.encoded_initial_state = self.encode_state(self.initial_state)

    def build_map_graph(self):
        """ build a graph of the map. """
        g = nx.grid_graph((self.map_shape[1], self.map_shape[0]))
        nodes_to_remove = []
        for node in g.nodes:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def encode_state(self, state):
        """
        Returns this state in encoded format (tuple of tuples).

        Parameters
        ----------
        state : dict
            The state to encode.

        Returns
        -------
        encoded_state : tuple
            The encoded state.
        """
        taxis = tuple(
            (
                taxi_name,
                state["taxis"][taxi_name]["location"],
                state["taxis"][taxi_name]["fuel"],
                state["taxis"][taxi_name]["capacity"]
            )
            for taxi_name in self.taxi_names)

        passengers = tuple(
            (
                passenger_name,
                state["passengers"][passenger_name]["location"],
                state["passengers"][passenger_name]["destination"]
            )
            for passenger_name in self.passengers_names)

        encoded_state = (taxis, passengers, state["turns to go"])

        return encoded_state

    def decode_state(self, state):
        """
        Return encoded state to dict format.

        Parameters
        ----------
        state : tuple
            The state to decode.

        Returns
        -------
        decoded_state : dict
            The decoded state.
        """
        decoded_state = {"taxis": {}, "passengers": {}}

        for taxi_name, location, fuel, capacity in state[0]:
            decoded_state["taxis"][taxi_name] = {"location": location,
                                                 "fuel": fuel,
                                                 "capacity": capacity}

        for passenger_name, location, destination in state[1]:
            decoded_state["passengers"][passenger_name] = {"location": location,
                                                           "destination": destination}

        decoded_state["turns to go"] = state[2]
        return decoded_state

    def copy_state(self, state):
        new_state = {"taxis": {}, "passengers": {}}
        for taxi_name in self.taxi_names:
            new_state["taxis"][taxi_name] = {"location": state["taxis"][taxi_name]["location"],
                                             "fuel": state["taxis"][taxi_name]["fuel"],
                                             "capacity": state["taxis"][taxi_name]["capacity"]}

        for p_name in self.passengers_names:
            new_state["passengers"][p_name] = {
                "location": state["passengers"][p_name]["location"],
                "destination": state["passengers"][p_name]["destination"]
            }
        new_state["turns to go"] = state["turns to go"]

        return new_state

    def is_action_legal(self, state, action):
        """
        Check if the action is legal to perform on the state.

        NOTICE!!!
        ------
        This function doesn't consider the remaining turns from the given state,
        it merely checks if the instructions are satisfied.

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to check.

        Returns
        -------
        bool
            If the action is legal - returns True, else - returns False.
        """

        def _is_move_action_legal(atomic_move_action):
            taxi_name = atomic_move_action[1]
            # validated at action creation
            if taxi_name not in self.taxi_names:
                return False
            # validated at action creation
            if state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = atomic_move_action[2]
            # validated at action creation
            return l2 in list(self.map_graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position - validated at action creation
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity - validated at action creation
            if state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            # check passenger is not in his destination - validated at action creation
            if state['passengers'][passenger_name]['destination'] == \
                    state['passengers'][passenger_name]['location']:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            # check same position (and that the passenger is on the taxi). - validated at action creation
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            if state['passengers'][passenger_name]['location'] == taxi_name and \
                    state['passengers'][passenger_name]['destination'] == state['taxis'][taxi_name]['location']:
                return True
            return False

        def _is_refuel_action_legal(refuel_action):
            # Check if taxi in gas station location. - validated at action creation
            taxi_name = refuel_action[1]
            i, j = state['taxis'][taxi_name]['location']
            if self.map[i][j] == 'G':
                return True
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

        if action == "reset" or action == "terminate":
            return True

        if len(action) != len(self.taxi_names):  # validated at action creation
            return False

        # for atomic_action in action:
        #
        #     # illegal move action - validated at action creation
        #     if atomic_action[0] == 'move':
        #         if not _is_move_action_legal(atomic_action):
        #             return False
        #
        #     # illegal pick action - validated at action creation
        #     elif atomic_action[0] == 'pick up':
        #         if not _is_pick_up_action_legal(atomic_action):
        #             return False
        #
        #     # illegal drop action - validated at action creation
        #     elif atomic_action[0] == 'drop off':
        #         if not _is_drop_action_legal(atomic_action):
        #             return False
        #
        #     # illegal refuel action - validated at action creation
        #     elif atomic_action[0] == 'refuel':
        #         if not _is_refuel_action_legal(atomic_action):
        #             return False
        #     # wait action - validated at action creation
        #     elif atomic_action[0] != 'wait':
        #         return False
        #
        # # check mutex action
        # if _is_action_mutex(action):
        #     return False

        # check taxis collision
        if len(state['taxis']) > 1:
            # create a mapping of taxi name to taxi location for all taxis in state
            taxis_location_dict = dict([(t, state['taxis'][t]['location']) for t in self.taxi_names])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]

            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                return False
        return True

    def possible_atomic_actions(self, state, taxi_name):
        """
        Return all possible actions that the given taxi can make in this state.

        Parameters
        ----------
        state : dict
            The state, in dict format.
        taxi_name : tuple
            The taxi's name.

        Returns
        -------
        possible_actions : list
            All atomic actions for the taxi.
        """
        possible_actions = []

        # "move" actions
        taxi_location = state["taxis"][taxi_name]["location"]
        row, column = taxi_location
        if state["taxis"][taxi_name]["fuel"] > 0:
            for loc in self.map_graph.neighbors(taxi_location):
                possible_actions.append(("move", taxi_name, loc))

        # "refuel" actions
        if self.map[row][column] == "G":
            possible_actions.append(("refuel", taxi_name))

        # "pick up" actions
        if state["taxis"][taxi_name]["capacity"] > 0:
            for passenger in self.passengers_names:
                if state["passengers"][passenger]["location"] == taxi_location and \
                        state["passengers"][passenger]["location"] != \
                        state["passengers"][passenger]["destination"]:
                    possible_actions.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for passenger in self.passengers_names:
            if state["passengers"][passenger]["location"] == taxi_name and \
                    state["passengers"][passenger]["destination"] == taxi_location:
                possible_actions.append(("drop off", taxi_name, passenger))

        possible_actions.append(("wait", taxi_name))
        return possible_actions

    def actions(self, state):
        """
        Returns all the actions that can be executed in the given state.
        The result should be a tuple (or other iterable) of actions,
        as defined in the problem instructions file.

        Parameters
        ----------
        state : dict
            The state, in dict format.

        Returns
        -------
        possible_action : tuple
            All atomic actions for the state.
        """

        if state["turns to go"] == 0:
            return []

        for action in filter(lambda a: self.is_action_legal(state, a),
                             product(*[self.possible_atomic_actions(state, taxi) for taxi in self.taxi_names])):
            yield action

        yield "reset"
        yield "terminate"

    def deterministic_outcome(self, state, action):
        """
        Returns the deterministic state that results from executing the given action on the given state.
        The action must be one of self.actions(state).

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to perform.

        Returns
        -------
        new_state : tuple | dict
            The resulting state.
        """

        if action == "terminate" or state["turns to go"] == 0:
            # in case of "terminate" or terminal state - no result, just end of game
            return None

        new_turns_to_go = state["turns to go"] - 1
        if action == "reset":
            new_state = self.copy_state(state=self.initial_state)

        else:
            new_state = self.copy_state(state=state)
            # a normal action
            for atomic_action in action:
                if atomic_action[0] == "move":
                    new_state["taxis"][atomic_action[1]]["location"] = atomic_action[2]
                    new_state["taxis"][atomic_action[1]]["fuel"] -= 1
                elif atomic_action[0] == "pick up":
                    new_state["passengers"][atomic_action[2]]["location"] = atomic_action[1]
                    new_state["taxis"][atomic_action[1]]["capacity"] -= 1
                elif atomic_action[0] == "drop off":
                    new_state["passengers"][atomic_action[2]]["location"] = \
                        new_state["taxis"][atomic_action[1]]["location"]
                    new_state["taxis"][atomic_action[1]]["capacity"] += 1
                elif atomic_action[0] == "refuel":
                    new_state["taxis"][atomic_action[1]]["fuel"] = self.initial_state["taxis"][atomic_action[1]]["fuel"]

        new_state["turns to go"] = new_turns_to_go
        return new_state

    def possible_outcomes(self, state, action):
        """
        Returns all possible states that result from executing the given action in the given state.
        The action must be one of self.actions(state).

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to perform.

        Returns
        -------
        new_state : tuple | dict
            The resulting state.
        """

        new_state = self.deterministic_outcome(state=state,
                                               action=action)

        if action == "terminate" or state["turns to go"] == 0:
            return []

        if action == "reset":
            return [new_state, ]

        possible_destinations = [list(set(self.possible_goals[p]).union(
            {new_state["passengers"][p]["destination"]})) for p in self.passengers_names]
        for destinations in product(*possible_destinations):

            for passenger, destination in zip(self.passengers_names, destinations):
                new_state["passengers"][passenger]["destination"] = destination

            yield self.copy_state(state=new_state)

    def reward(self, action):
        """ The function that calculates reward of performing this action on a state. """
        if action == "reset":
            return -RESET_PENALTY
        elif action == "terminate":
            return 0.0
        score = 0.0
        for atomic_action in action:
            if atomic_action[0] == "drop off":
                score += DROP_IN_DESTINATION_REWARD
            elif atomic_action[0] == "refuel":
                score -= REFUEL_PENALTY
        return score

    def transition(self, state, action, new_state):
        """
        The function that calculates the probability of getting to new_state from state by performing action.

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to perform.
        new_state : dict
            possible new state, in dict format.

        Returns
        -------
        probability : float
            The probability to reach new_state from state using action.
        """

        if action == "reset":
            # when calling action "reset" there is only 1 possible result, which is the initial state,
            # with turns to go < initial_state["turns to go"].
            encoded_new_state = self.encode_state(state=new_state)
            return float(self.encoded_initial_state[0] == encoded_new_state[0] and
                         self.encoded_initial_state[1] == encoded_new_state[1] and
                         self.encoded_initial_state[2] > encoded_new_state[2])
        else:

            # if calling a normal action, then we actually have something to calculate.
            outcome = self.deterministic_outcome(state=state,
                                                 action=action)
            if outcome is None:
                # when calling action "terminate", or when calling on a terminal state,
                # the outcome will be None and the game terminates, thus the probability to arrive at any state is 0.
                return 0.0

            probability = 1.0
            for passenger in self.passengers_names:
                # represented conditions as boolean expressions:
                pc = self.prob_change_goal[passenger]
                destination = outcome["passengers"][passenger]["destination"]
                new_destination = new_state["passengers"][passenger]["destination"]
                possible_goals = self.possible_goals[passenger]
                lp = len(possible_goals)
                ind1 = destination in possible_goals
                ind2 = destination == new_destination

                # original:
                # if ind1:
                #     probability *= (pc / len(possible_goals)) + ind2 * (1 - pc)
                # else:
                #     if ind2:
                #         probability *= 1 - pc
                #     else:
                #         probability *= (pc / len(possible_goals))

                # simplified mathematical formula on paper and got:
                # probability *= ind2 * (1 - pc - (pc / lp)) + (1 + (ind1 * ind2)) * (pc / lp)

                # written as conditions we have:
                if ind2:
                    probability *= 1 - pc + (ind1 * (pc / lp))
                else:
                    probability *= pc / lp
            return probability

    def act(self, state):
        """ The Agent's policy function. """
        raise NotImplemented


class OptimalTaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        # self.game_graph = nx.DiGraph()
        # self.build_game_graph_recursive(state=self.initial_state)
        # self.value_iteration(state=self.initial_state)
        self.states = self.all_state_permutations(self.initial_state)
        self.filter_states(states=self.states, initial_state=self.encoded_initial_state)
        # self.basic_policy_iteration(states=self.states)
        self.value_iteration(states=self.states)
        print("any value? ", all(self.states[n]["value"] == 0 for n in self.states))
        print("initial state value: ", self.states[self.encoded_initial_state]['value'])

    def is_terminal_state(self, state):
        # in a terminal state there are no more possible actions to do!
        return state["turns to go"] == 0

    def expand(self, state):
        for action in self.actions(state=state):
            reward = self.reward(action=action)
            for outcome in self.possible_outcomes(state=state,
                                                  action=action):
                yield action, outcome, reward

    def build_game_graph_recursive(self, state):
        """
        Build a game DAG represented as a NetworkX DiGraph, starting from the initial state.
        The function uses a depth-first search algorithm to explore the DAG.

        Parameters
        ----------
        state : dict
            The root state of the graph.

        Returns
        -------
        Node : tuple
            The root node of the game graph.
        """
        if self.is_terminal_state(state=state):
            # If a state is terminal we will not add it to the graph.
            return None
        # If the state is not terminal, then we add it to the graph.
        node = self.encode_state(state=state)
        self.game_graph.add_node(node_for_adding=node,
                                 value=float("-inf"),
                                 policy=None)
        # For every child state of this state:
        for action, child_state, reward in self.expand(state=state):
            # Explore the subgraph rooted at this child state, and return its node in the graph.
            child_node = self.build_game_graph_recursive(state=child_state)
            # If the returned node is None, then this child is a terminal state, thus we will not add it to the graph
            if child_node is not None:
                # Else, we connect the subgraph rooted at this child node to this node as a successor of this node.
                self.game_graph.add_edge(u_of_edge=node,
                                         v_of_edge=child_node,
                                         action=action,
                                         reward=reward)
        # Finally, return this node to complete the recursion.
        return node

    def build_game_graph_iterative(self):
        """
        Build a game DAG represented as a NetworkX DiGraph, starting from the initial state.
        The function uses a depth-first search algorithm to explore the DAG.

        Parameters
        ----------
            None.

        Returns
        -------
        Node : tuple
            The root node of the game graph.
        """
        # Initialize an empty graph and a stack for storing nodes that need to be explored.
        self.game_graph = nx.DiGraph()
        stack = deque()

        # Add the initial state of the game as the root node of the graph, and push it onto the stack.
        stack.append(self.initial_state)

        # While the stack is not empty:
        while stack:
            # Pop a node from the top of the stack.
            state = stack.pop()
            # The state is not terminal, so we add it to the graph.
            node = self.encode_state(state=state)
            if node not in self.game_graph:
                self.game_graph.add_node(node_for_adding=node,
                                         value=float("-inf"),
                                         policy=None)
            # For every child state of this state:
            for action, child_state, reward in self.expand(state=state):
                # If the child_node is a terminal state, we will not add it to the graph.
                if not self.is_terminal_state(state=child_state):
                    stack.append(child_state)
                    child_node = self.encode_state(state=child_state)
                    # Else, we connect the subgraph rooted at this child node to this node as a successor of this node.
                    if child_node not in self.game_graph:
                        self.game_graph.add_node(node_for_adding=child_node,
                                                 value=float("-inf"),
                                                 policy=None)

                    self.game_graph.add_edge(u_of_edge=node,
                                             v_of_edge=child_node,
                                             action=action,
                                             reward=reward)

    def value_iteration(self, states):
        """
        Perform value-iteration on the game graph and update the "value" and "policy" attributes with the result.

        Parameters
        ----------
        states : dict
            The state to calculate the value for.

        Returns
        -------
        value : float
            the value of the given state.
        """
        # Initialize the value of every state to a "neutral" value
        for state in states:
            states[state]["value"] = 0.0

        # While the value of any state has changed
        value_changed = True
        while value_changed:
            value_changed = False
            # For each state, calculate the value of the state by taking the maximum of the expected rewards of all actions
            for state in states:
                decoded_state = self.decode_state(state)
                old_value = states[state]["value"]  # Store the old value of the state
                best_value = float("-inf")  # Initialize the best value to a very low number
                best_policy = "terminate"  # Initialize the best policy to None
                # For each action that can be taken from the state
                for action in self.actions(decoded_state):
                    value = 0.0  # Initialize the value of the action to 0
                    # For each resulting state
                    for new_state in self.possible_outcomes(decoded_state, action):
                        encoded_new_state = self.encode_state(new_state)
                        prob = self.transition(decoded_state, action,
                                               new_state)  # Calculate the probability of transitioning to the new state
                        value += prob * states[encoded_new_state][
                            "value"]  # Calculate the expected reward by summing the rewards of all transitions from this state to other states, weighted by their probabilities
                    value += self.reward(action)  # Add the immediate reward for taking the action
                    # If the value of the action is greater than the current best value
                    if value > best_value:
                        best_value = value  # Update the best value
                        best_policy = action  # Update the best policy
                states[state]["value"] = best_value  # Set the value of the state to the best value
                states[state]["policy"] = best_policy  # Set the policy of the state to the best policy
                # If the value of the state has changed
                if best_value != old_value:
                    value_changed = True  # Set the value_changed flag to True to continue the iteration

    def all_state_permutations(self, state):
        """
        Receives an initial state and returns all possible permutations of it,
        regardless of the possibility of them occurring.

        Parameters
        ----------
        state : dict
            The initial state, in dict format.

        Returns
        -------
        states : dict
            A dict with all the possible states in its keys.
        """
        encoded_state = self.encode_state(state=state)
        states = {encoded_state: {"policy": "terminate", "value": 0.0}, }

        state_p_tuples = []
        for p_name in self.passengers_names:
            p_tuples = []
            dests = list(set(self.possible_goals[p_name]).union(
                {state["passengers"][p_name]["location"], state["passengers"][p_name]["destination"]}))
            locs = [*dests, *self.taxi_names]
            for loc, dest in product(locs, dests):
                p_tuples.append((p_name, loc, dest))
            state_p_tuples.append(p_tuples)
        passengers = list(product(*state_p_tuples))

        state_taxi_tuples = []
        taxi_locs = list(self.map_graph.nodes)
        for taxi_name in self.taxi_names:
            taxi_tuples = []
            fuels = list(range(state["taxis"][taxi_name]["fuel"], -1, -1))
            caps = list(range(state["taxis"][taxi_name]["capacity"], -1, -1))
            for loc, fuel, cap in product(taxi_locs, fuels, caps):
                taxi_tuples.append((taxi_name, loc, fuel, cap))
            state_taxi_tuples.append(taxi_tuples)
        taxis = list(product(*state_taxi_tuples))
        turns_to_go = list(range(encoded_state[2] - 1, -1, -1))

        for taxis_tuple, passengers_tuple, turns in product(taxis, passengers, turns_to_go):
            states[(taxis_tuple, passengers_tuple, turns)] = {"policy": "terminate", "value": 0.0}

        return states

    def filter_states(self, states, initial_state):
        """
        Returns only the states that are reachable from the initial state.

        Parameters
        ----------
        states : dict
            All states in the game.
        initial_state : tuple
            The initial state of the game (encoded).

        Returns
        -------
        None
            The filter is inplace.
        """
        queue = [initial_state]
        visited = set()
        while queue:
            state = queue.pop(0)
            visited.add(state)
            decoded_state = self.decode_state(state)
            for action in self.actions(decoded_state):
                for new_state in self.possible_outcomes(decoded_state, action):
                    encoded_new_state = self.encode_state(new_state)
                    if encoded_new_state not in visited:
                        visited.add(encoded_new_state)
                        queue.append(encoded_new_state)

        # Remove any states that are not in the visited set
        for state in list(states.keys()):
            if state not in visited:
                del states[state]

    def basic_policy_iteration(self, states):
        """
        Performs policy iteration on the states given in this dict.

        Parameters
        ----------
        states : dict
            All the possible states in the game.

        Returns
        -------
        None
            The output is stored inplace.
        """
        # Initialize the policy for every state to a random action
        for state in states:
            if state[2]:
                states[state]["policy"] = random.choice(list(self.actions(self.decode_state(state))))
            else:
                states[state]["policy"] = "terminate"

        policy_changed = True
        while policy_changed:
            policy_changed = False

            # Calculate the value of each state using the current policy
            for state in states:
                decoded_state = self.decode_state(state)
                value = 0
                for new_state in self.possible_outcomes(decoded_state, states[state]["policy"]):
                    encoded_new_state = self.encode_state(new_state)
                    prob = self.transition(decoded_state, states[state]["policy"], new_state)
                    value += prob * states[encoded_new_state]["value"]
                value += self.reward(states[state]["policy"])
                states[state]["value"] = value

            # Set the policy for each state to the action that maximizes the value of the state
            for state in states:
                decoded_state = self.decode_state(state)
                old_policy = states[state]["policy"]
                best_policy = "terminate"
                best_value = 0.0
                for action in self.actions(decoded_state):
                    value = 0
                    for new_state in self.possible_outcomes(decoded_state, action):
                        encoded_new_state = self.encode_state(new_state)
                        prob = self.transition(decoded_state, action, new_state)
                        value += prob * states[encoded_new_state]["value"]
                    value += self.reward(action)
                    if value > best_value:
                        best_value = value
                        best_policy = action
                states[state]["policy"] = best_policy
                if best_policy != old_policy:
                    policy_changed = True

    def basic_bitch_policy(self, state):
        """
        The most basicest bitchest of policies!

        Parameters
        ----------
        state : dict
            The state to calculate the policy for.

        Returns
        -------
        action : tuple
            The policy.
        """
        atomic_actions = []
        for taxi_name in self.taxi_names:
            for p in self.passengers_names:
                if state["passengers"][p]["location"] == taxi_name and \
                        state["taxis"][taxi_name]["location"] == state["passengers"][p]["destination"]:
                    atomic_actions.append(("drop off", taxi_name, p))
                elif state["passengers"][p]["location"] == state["taxis"][taxi_name]["location"] and \
                        state["passengers"][p]["location"] != state["passengers"][p]["destination"]:
                    atomic_actions.append(("pick up", taxi_name, p))
                else:
                    path_to_p = nx.shortest_path(G=self.map_graph,
                                                 source=state["taxis"][taxi_name]["location"],
                                                 target=state["passengers"][p]["location"])
                    if len(path_to_p) < state["taxis"][taxi_name]["fuel"]:
                        atomic_actions.append(("move", taxi_name, path_to_p[1]))
                    else:


    def act(self, state):
        return self.states[self.encode_state(state)]["policy"]


class TaxiAgent(Agent):
    def __init__(self, initial):
        Agent.__init__(self, initial)
        self.default_agent = OptimalTaxiAgent(initial)

    def act(self, state):
        return self.default_agent.act(state)
