import random
import numpy as np
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

        # optimization related variables:
        self.outcome = self.copy_state(self.initial_state, to_outcome=False)

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

    def copy_state(self, state, to_outcome=True):
        if to_outcome:
            new_state = self.outcome
        else:
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
            return

        yield from filter(lambda a: self.is_action_legal(state, a),
                          product(*[self.possible_atomic_actions(state, taxi) for taxi in self.taxi_names]))
        yield "reset"
        yield "terminate"

    def deterministic_outcome(self, state, action, to_outcome=True):
        """
        Returns the deterministic state that results from executing the given action on the given state.
        The action must be one of self.actions(state).

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to perform.
        to_outcome : bool
            If True then set output into outcome attribute, else set into copy.

        Returns
        -------
        new_state : tuple | dict
            The resulting state.
        """

        if action == "terminate" or state["turns to go"] == 0:
            # in case of "terminate" or terminal state - no result, just end of game.
            return None

        new_turns_to_go = state["turns to go"] - 1
        if action == "reset":
            new_state = self.copy_state(state=self.initial_state, to_outcome=to_outcome)

        else:
            new_state = self.copy_state(state=state, to_outcome=to_outcome)
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

    def possible_outcomes(self, state, action, to_outcome=True):
        """
        Returns all possible states that result from executing the given action in the given state.
        The action must be one of self.actions(state).

        Parameters
        ----------
        state : dict
            The state, in dict format.
        action : tuple
            The action to perform.
        to_outcome : bool
            If True then copy to ~self.outcome~ attribute, else copy to new object.

        Returns
        -------
        new_state : tuple | dict
            The resulting state.
        """

        if action == "terminate" or state["turns to go"] == 0:
            return

        new_state = self.deterministic_outcome(state=state,
                                               action=action,
                                               to_outcome=to_outcome)
        if action == "reset":
            yield new_state
        else:
            possible_destinations = [list(set(self.possible_goals[p]).union(
                {new_state["passengers"][p]["destination"]})) for p in self.passengers_names]

            for destinations in product(*possible_destinations):
                for passenger, destination in zip(self.passengers_names, destinations):
                    new_state["passengers"][passenger]["destination"] = destination

                yield self.copy_state(new_state, to_outcome=to_outcome)

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
            if action == "terminate" or state["turns to go"] == 0:
                # when calling action "terminate", or when calling on a terminal state,
                # the outcome will be None and the game terminates, thus the probability to arrive at any state is 0.
                return 0.0

            probability = 1.0
            for passenger in self.passengers_names:
                # represented conditions as boolean expressions:
                pc = self.prob_change_goal[passenger]
                destination = state["passengers"][passenger]["destination"]
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
    def __init__(self, initial, threshold=float("inf")):
        Agent.__init__(self, initial)
        self.gas_stations = [(i, j)
                             for i in range(self.map_shape[0])
                             for j in range(self.map_shape[1])
                             if self.map[i][j] == "G"]
        self.states = self.all_state_permutations(self.initial_state)
        self.filter_states(states=self.states, initial_state=self.encoded_initial_state)
        self.policy_iteration_with_threshold(states=self.states, threshold=threshold)

    def is_terminal_state(self, state):
        # in a terminal state there are no more possible actions to do!
        return state["turns to go"] == 0

    def expand(self, state):
        for action in self.actions(state=state):
            reward = self.reward(action=action)
            for outcome in self.possible_outcomes(state=state,
                                                  action=action):
                yield action, outcome, reward

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

    def calculate_value(self, state, action):
        value = 0.0
        for new_state in self.possible_outcomes(state, action):
            encoded_new_state = self.encode_state(new_state)
            prob = self.transition(state, action, new_state)
            value += prob * self.states[encoded_new_state]["value"]
        value += self.reward(action)
        return value

    def policy_iteration_with_threshold(self, states, threshold=float("inf")):
        """
        Performs policy iteration on the states given in this dict.

        Parameters
        ----------
        states : dict
            All the possible states in the game.
        threshold : float
            A minimal value for the expected score that the policy should have.
            For example, if set to be 0.0,
            than will stop improving once a policy that has a positive expectation is achieved

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
                states[state]["value"] = self.calculate_value(decoded_state, states[state]["policy"])

            # Set the policy for each state to the action that maximizes the value of the state
            for state in states:
                decoded_state = self.decode_state(state)
                old_policy = states[state]["policy"]
                best_policy = "terminate"
                best_value = 0.0
                for action in self.actions(decoded_state):
                    value = self.calculate_value(decoded_state, action)
                    if value > best_value:
                        best_value = value
                        best_policy = action
                states[state]["policy"] = best_policy
                if best_policy != old_policy:
                    policy_changed = True
                if states[self.encoded_initial_state]["value"] > threshold:
                    break

    def basic_bitch_policy(self, state):
        """
        The most basicest and bitchest of policies!

        Parameters
        ----------
        state : dict
            The state to calculate the policy for.

        Returns
        -------
        action : tuple
            The policy.
        """
        if not state["turns to go"]:
            return "terminate"
        atomic_actions = []
        reset_flag = False
        for taxi_name in self.taxi_names:
            for p in self.passengers_names:
                if state["passengers"][p]["location"] == taxi_name and \
                        state["taxis"][taxi_name]["location"] == state["passengers"][p]["destination"]:
                    atomic_actions.append(("drop off", taxi_name, p))
                elif state["passengers"][p]["location"] == state["taxis"][taxi_name]["location"] and \
                        state["passengers"][p]["location"] != state["passengers"][p]["destination"]:
                    atomic_actions.append(("pick up", taxi_name, p))
                else:
                    path_to_p = list(nx.shortest_path(G=self.map_graph,
                                                      source=state["taxis"][taxi_name]["location"],
                                                      target=state["passengers"][p]["location"]))
                    if len(path_to_p) < state["taxis"][taxi_name]["fuel"]:
                        atomic_actions.append(("move", taxi_name, path_to_p[1]))
                    else:
                        nearest_gas_station = self.gas_stations[0]
                        min_dist = list(nx.shortest_path_length(G=self.map_graph,
                                                                source=state["taxis"][taxi_name]["location"],
                                                                target=nearest_gas_station))
                        for gas_station in self.gas_stations:
                            dist = nx.shortest_path_length(G=self.map_graph,
                                                           source=state["taxis"][taxi_name]["location"],
                                                           target=gas_station)
                            if dist < min_dist:
                                nearest_gas_station = gas_station
                                min_dist = dist
                        if min_dist > state["taxis"][taxi_name]["fuel"]:
                            reset_flag = True
                        else:
                            if not min_dist:
                                atomic_actions.append(("refuel", taxi_name))
                            else:
                                path_to_g = nx.shortest_path(G=self.map_graph,
                                                             source=state["taxis"][taxi_name]["location"],
                                                             target=nearest_gas_station)
                                atomic_actions.append(("move", taxi_name, path_to_g[1]))
        if reset_flag:
            return "reset"
        else:
            return tuple(atomic_actions)

    def act(self, state):
        return self.states[self.encode_state(state)]["policy"]


class QLearningAgent(Agent):
    def __init__(self, initial, explore_policy="glie1"):
        Agent.__init__(self, initial)
        self.state = self.copy_state(self.initial_state)
        self.score = 0
        self.num_features = len(self.calculate_features(self.state, list(self.actions(self.state))[0]))
        self.weights = self.initialize_weights()
        self.alpha = 1
        self.temperature = 1
        self.explore_policy = explore_policy  # The explore/exploit policy
        self.num_visits = 0  # used for the other explore/exploit policy

        total_episodes = 0
        while self.score <= 0:
            total_episodes += 1
            print(f"\nEpisode Number {total_episodes}:")
            self.episode_simulation()
        print(f"final episode score: {self.score}")
        print(f"number of episodes: {total_episodes}")

    def episode_simulation(self):
        self.state = self.copy_state(self.initial_state)
        self.num_visits = 0
        self.score = 0
        for i in range(self.initial_state["turns to go"]):

            # print(i, self.state["turns to go"])
            # Choose an action using the current weights
            action = self.choose_action(self.state, explore=True)
            print(action)
            # Get the reward for the transition from state to new_state via action
            reward = self.reward(action)
            self.score += reward
            # Get all possible outcomes of taking the action
            outcomes = list(self.possible_outcomes(self.state, action, to_outcome=False))
            if not outcomes:
                break
            # print(action)
            # for o in outcomes:
            #     print(o)

            # Get the probabilities of each outcome
            probabilities = []
            for outcome in outcomes:
                probabilities.append(self.transition(self.state, action, outcome))
            # print(f"probabilities: {probabilities}")

            # Choose new state based on the probabilities of each outcome
            outcomes_indices = list(range(len(outcomes)))
            new_state = outcomes[np.random.choice(outcomes_indices, p=probabilities)]

            # Update the weights based on the result of the action
            self.update(self.state, action, new_state, reward, self.alpha)

            # Set the current state to the new state
            self.state = new_state

    def initialize_weights(self):
        initial_weights = [1, ]
        for taxi in self.taxi_names:
            initial_weights.append(0)
            initial_weights.append(0)
            initial_weights.append(0)
            initial_weights.append(1)
            initial_weights.append(1)
        for p in self.passengers_names:
            initial_weights.append(0)
            initial_weights.append(0)
            initial_weights.append(0)
            initial_weights.append(0)
            initial_weights.append(1)
            initial_weights.append(1)
            initial_weights.append(2)
            initial_weights.append(0)
        initial_weights.append(0)

        initial_weights.append(-1000)
        initial_weights.append(-1000)
        for _ in self.taxi_names:
            initial_weights.append(1)  # move
            initial_weights.append(1)  # move
            initial_weights.append(1)  # move
            initial_weights.append(1)  # move
            for _ in self.passengers_names:
                initial_weights.append(1)  # good move (take passenger closer to his destination)
            initial_weights.append(0)
            initial_weights.append(1)  # pick up
            initial_weights.append(1)  # drop off
            initial_weights.append(-1)  # refuel
            initial_weights.append(-1)  # wait
        initial_weights.append(0)  # for bias term
        return np.array(initial_weights, dtype=float)

    def get_nearest_passenger(self, state, taxi):
        t_loc = state["taxis"][taxi]["location"]
        min_dist = float("inf")
        min_p = None
        for p in self.passengers_names:
            if state["passengers"][p]["location"] == state["passengers"][p]["destination"]:
                continue
            p_loc = state["passengers"][p]["location"]
            if isinstance(p_loc, str):
                if p_loc != taxi:
                    continue
                else:
                    p_loc = state["taxis"][p_loc]["location"]
            dist = abs(t_loc[0] - p_loc[0]) + abs(t_loc[1] - p_loc[1])
            if min_dist > dist:
                min_dist = dist
                min_p = p
        if min_dist == float("inf"):
            min_dist = 0
            min_p = self.passengers_names[0]
        return min_dist, min_p

    def calculate_features(self, state, action):
        features = [state["turns to go"], ]
        for taxi in self.taxi_names:
            features.append(self.get_nearest_passenger(state, taxi)[0])
            features.append(state["taxis"][taxi]["location"][0])
            features.append(state["taxis"][taxi]["location"][1])
            features.append(state["taxis"][taxi]["fuel"])
            features.append(state["taxis"][taxi]["capacity"])

        p_in_dest = 0
        for p in self.passengers_names:
            if isinstance(state["passengers"][p]["location"], str):
                location = state["taxis"][state["passengers"][p]["location"]]["location"]
            else:
                location = state["passengers"][p]["location"]
            features.append(location[0])
            features.append(location[1])
            features.append(state["passengers"][p]["destination"][0])
            features.append(state["passengers"][p]["destination"][1])
            x_dist = abs(state["passengers"][p]["destination"][0] - location[0])
            y_dist = abs(state["passengers"][p]["destination"][1] - location[1])
            features.append(x_dist)
            features.append(y_dist)
            features.append(x_dist + y_dist)
            if location == state["passengers"][p]["destination"]:
                features.append(1)
                p_in_dest += 1
            else:
                features.append(0)
        features.append(p_in_dest)

        if isinstance(action, str):
            if action == "terminate":
                features.append(1)
            else:
                features.append(0)
            if action == "reset":
                features.append(1)
            else:
                features.append(0)
            for _ in self.taxi_names:
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                for _ in self.passengers_names:
                    features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
        else:
            features.append(0)
            features.append(0)
            for atomic_action in action:
                if atomic_action[0] == "move":
                    next_location = atomic_action[2]
                    current_location = state["taxis"][atomic_action[1]]["location"]
                    for loc in [(current_location[0] - 1, current_location[1]),
                                (current_location[0] + 1, current_location[1]),
                                (current_location[0], current_location[1] - 1),
                                (current_location[0], current_location[1] + 1)]:
                        if next_location == loc:
                            features.append(1)
                        else:
                            features.append(0)

                    for p in self.passengers_names:
                        if state["passengers"][p]["location"] == atomic_action[1]:
                            dest = state["passengers"][p]["destination"]
                            if abs(dest[0] - next_location[0]) + abs(dest[1] - next_location[1]) <= \
                                    abs(dest[0] - current_location[0]) + abs(dest[1] - current_location[1]):
                                p_val = 1
                            else:
                                p_val = -1
                        else:
                            p_val = 0
                        features.append(p_val)

                    current_dist, nearest_p = self.get_nearest_passenger(state, atomic_action[1])
                    p_loc = state["passengers"][nearest_p]["location"]
                    if isinstance(p_loc, str):
                        p_loc = state["taxis"][p_loc]["location"]
                    next_dist = abs(next_location[0] - p_loc[0]) + abs(next_location[1] - p_loc[1])
                    features.append(current_dist - next_dist)
                else:
                    features.append(0)
                    features.append(0)
                    features.append(0)
                    features.append(0)
                    features.append(0)
                    for p in self.passengers_names:
                        features.append(0)
                if atomic_action[0] == "pick up":
                    features.append(1)
                else:
                    features.append(0)
                if atomic_action[0] == "drop off":
                    features.append(1)
                else:
                    features.append(0)
                if atomic_action[0] == "refuel":
                    features.append(1)
                else:
                    features.append(0)
                if atomic_action[0] == "wait":
                    features.append(1)
                else:
                    features.append(0)

        features.append(1)  # for bias term

        return np.array(features)

    def calculate_q_value(self, state, action):
        return np.dot(self.calculate_features(state, action), self.weights)

    def choose_action(self, state, explore=False):
        # Calculate the value of the state by taking the dot product of the features and weights of the state
        actions = list(self.actions(state))
        # actions.remove("reset")
        # actions.remove("terminate")
        actions_indices = list(range(len(actions)))

        # If the explore flag is set to True
        if explore:
            actions.remove("reset")
            actions.remove("terminate")
            actions_indices = list(range(len(actions)))
            self.num_visits += 1

            # Choose an action using the specified policy
            if self.explore_policy == "boltzmann":

                # Apply the Boltzmann exploration policy
                # Use a temperature that decreases with time:
                self.temperature *= 0.999
                # temperature = 1 / self.num_visits
                values = np.exp(
                    np.array([self.calculate_q_value(state, action) / self.temperature for action in actions]))

                # Calculate the Boltzmann probabilities
                boltzmann_probs = values / np.sum(values)
                # print(f"weights: {self.weights}")
                # print(f"values: {values}")
                # print(f"boltzmann_probs: {boltzmann_probs}")

                # Choose an action based on the Boltzmann probabilities
                return actions[np.random.choice(actions_indices, p=boltzmann_probs)]

            elif self.explore_policy == "glie1":
                # Apply GLIE policy 1 with p(t)=1 / num_visits

                # Calculate the GLIE exploration rate
                epsilon = 1 / self.num_visits
                if np.random.rand() < epsilon:
                    return actions[np.random.choice(actions_indices)]
            else:
                raise ValueError("Invalid policy specified")

        # Exploit by choosing the action with the highest Q value
        q_values = [self.calculate_q_value(state, action) for action in actions]
        return actions[np.argmax(q_values)]

    def update(self, s, a, s_prime, r, alpha):
        # Update the weights based on the result of the action
        features = self.calculate_features(s, a)
        if s_prime["turns to go"] > 0:
            max_q_s_a_prime, max_a_prime = max([(self.calculate_q_value(s_prime, a_prime), a_prime)
                                                for a_prime in self.actions(s_prime)],
                                               key=lambda t: t[0])
            # print(max_q_s_a_prime, max_a_prime)
        else:
            max_q_s_a_prime = 0
        f = self.shaped_reward(s, s_prime)
        target = r + f + max_q_s_a_prime
        error = target - self.calculate_q_value(s, a)
        self.weights += alpha * error * features
        self.weights = self.weights / np.sum(self.weights)

    def potential(self, state):
        e = 1e-6
        total_dist_to_dest = 0
        for p in self.passengers_names:
            p_loc = state["passengers"][p]["location"]
            p_dest = state["passengers"][p]["destination"]
            if isinstance(p_loc, str):
                total_dist_to_dest += 1 / len(self.passengers_names)
                p_loc = state["taxis"][p_loc]["location"]
            total_dist_to_dest += abs(p_dest[0] - p_loc[0]) + abs(p_dest[1] - p_loc[1])

        total_dist_to_p = 0
        for t in self.taxi_names:
            t_loc = state["taxis"][t]["location"]
            min_dist = float("inf")
            for p in self.passengers_names:
                p_loc = state["passengers"][p]["location"]
                if isinstance(p_loc, str):
                    total_dist_to_dest += 1 / len(self.passengers_names)
                    if p_loc != t:
                        continue
                    else:
                        p_loc = state["taxis"][p_loc]["location"]
                dist = abs(t_loc[0] - p_loc[0]) + abs(t_loc[1] - p_loc[1])
                if min_dist > dist:
                    min_dist = dist
            total_dist_to_p += min_dist
        r1 = 1 - ((total_dist_to_dest + e) / (len(self.passengers_names) * (sum(self.map_shape) - 2 + e)))
        r2 = 1 - ((total_dist_to_p + e) / (len(self.taxi_names) * (sum(self.map_shape) - 2 + e)))
        r3 = state["turns to go"] / self.initial_state["turns to go"]
        return (r1 + r2 + r3) / 3

    def shaped_reward(self, state, new_state):
        """ The function that calculates reward of performing this action on a state. """
        return self.potential(new_state) - self.potential(state)

    def act(self, state):
        self.state = state
        # Calculate the best action for the given state using the pre-computed weights
        policy = max([action for action in self.actions(state)],
                     key=lambda a: self.calculate_q_value(state, a))
        # print(f"policy: {policy}")
        return policy


class TaxiAgent(OptimalTaxiAgent):
    def __init__(self, initial):
        super().__init__(initial, threshold=0)

    def act(self, state):
        return super().act(state)
