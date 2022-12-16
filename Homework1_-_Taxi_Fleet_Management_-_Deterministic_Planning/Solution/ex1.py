import search
import random
import math
from itertools import product
import networkx as nx
import stockfish

ids = ["322691718", "313539850"]

NOT_PICKED_UP = 0
PICKED_UP = 1
DELIVERED = 2


def manhattan(p, q):
    return math.fsum(math.fabs(x - y) for x, y in zip(p, q))


def get_graph(map: list[list[str]], map_shape: tuple) -> nx.Graph:
    g = nx.Graph()
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            if map[i][j] != "I":
                g.add_node((i, j), is_gas=(map[i][j] == "G"))
                for k, l in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= k < map_shape[0] and 0 <= l < map_shape[1]:
                        if map[k][l] != "I":
                            g.add_edge((i, j), (k, l))
    return g


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map_shape = (len(initial["map"]), len(initial["map"][0]))
        self.graph = get_graph(initial["map"], self.map_shape)
        self.gas_stations = [(i, j)
                             for i in range(self.map_shape[0])
                             for j in range(self.map_shape[1])
                             if initial["map"][i][j] == "G"]

        self.unsolvable = False
        for passenger in initial["passengers"].keys():
            loc = initial["passengers"][passenger]["location"]
            dest = initial["passengers"][passenger]["destination"]
            if initial["map"][loc[0]][loc[1]] == "I" or initial["map"][dest[0]][dest[1]] == "I":
                self.unsolvable = True
                break

        self.taxi_names = list(initial["taxis"].keys())
        self.n_taxis = len(self.taxi_names)
        self.fleet_capacity = sum([initial["taxis"][taxi]["capacity"] for taxi in self.taxi_names])
        for taxi_name in self.taxi_names:
            initial["taxis"][taxi_name]["traveling"] = []
            initial["taxis"][taxi_name]["fuel_capacity"] = initial["taxis"][taxi_name]["fuel"]

        self.passengers_names = list(initial["passengers"].keys())
        self.n_passengers = len(self.passengers_names)
        self.passengers_status = {}
        for passenger in self.passengers_names:
            initial["passengers"][passenger]["status"] = NOT_PICKED_UP
            self.passengers_status[passenger] = NOT_PICKED_UP

        self.state = initial
        initial = self.hash_state()
        search.Problem.__init__(self, initial)

    def graph_distance(self, p, q):
        return nx.shortest_path_length(G=self.graph, source=p, target=q)

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

    def get_available_moves(self, taxi_name):
        """
        Return all possible moves that this taxi can make.
        :param taxi_name: name of taxi.
        :return: list of moves for this taxi.
        """
        available_moves = []

        # "move" actions
        row = self.state["taxis"][taxi_name]["location"][0]
        column = self.state["taxis"][taxi_name]["location"][1]
        if self.state["taxis"][taxi_name]["fuel"] > 0:
            if row > 0 and self.state["map"][row - 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row - 1, column)))
            if row < self.map_shape[0] - 1 and self.state["map"][row + 1][column] != 'I':
                available_moves.append(("move", taxi_name, (row + 1, column)))
            if column > 0 and self.state["map"][row][column - 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column - 1)))
            if column < self.map_shape[1] - 1 and self.state["map"][row][column + 1] != 'I':
                available_moves.append(("move", taxi_name, (row, column + 1)))

        # "refuel" actions
        if self.state["map"][row][column] == "G":
            available_moves.append(("refuel", taxi_name))

        # "pick up" actions
        if len(self.state["taxis"][taxi_name]["traveling"]) < self.state["taxis"][taxi_name]["capacity"]:
            for passenger in self.passengers_names:
                if self.state["passengers"][passenger]["location"][0] == row and \
                        self.state["passengers"][passenger]["location"][1] == column:
                    available_moves.append(("pick up", taxi_name, passenger))

        # "drop off" actions
        for traveller in self.state["taxis"][taxi_name]["traveling"]:
            if self.state["passengers"][traveller]["destination"][0] == row and \
                    self.state["passengers"][traveller]["destination"][1] == column:
                available_moves.append(("drop off", taxi_name, traveller))

        available_moves.append(("wait", taxi_name))
        return available_moves

    def is_legal(self, action, taxis_location: dict):
        """
        check if an action is legal in a given state, so that no 2 taxis end up in the same area.
        :param taxis_location: dict of taxi:location pairs
        :param action: action to check
        :return: True if legal, else False
        """
        for taxi_action in action:
            if taxi_action[0] == "move":
                taxis_location[taxi_action[1]] = taxi_action[2]

        if len(set(taxis_location.values())) < self.n_taxis:
            return False
        return True

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        if self.unsolvable:
            return
        self.unhash_state(state)
        taxi_loc = dict((taxi_name, taxi_values["location"]) for taxi_name, taxi_values in self.state["taxis"].items())

        moves = []
        for taxi in self.taxi_names:
            moves.append(self.get_available_moves(taxi))

        for action in product(*moves):
            if self.is_legal(action, taxis_location=taxi_loc):
                yield action

    def result(self, state, action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state). """
        self.unhash_state(state)
        for taxi_action in action:
            if taxi_action[0] == "move":
                self.state["taxis"][taxi_action[1]]["location"] = taxi_action[2]
                self.state["taxis"][taxi_action[1]]["fuel"] -= 1
                for traveller in self.state["taxis"][taxi_action[1]]["traveling"]:
                    self.state["passengers"][traveller]["location"] = taxi_action[2]
            elif taxi_action[0] == "pick up":
                self.state["taxis"][taxi_action[1]]["traveling"].append(taxi_action[2])
                self.state["passengers"][taxi_action[2]]["status"] = PICKED_UP
            elif taxi_action[0] == "drop off":
                self.state["taxis"][taxi_action[1]]["traveling"].remove(taxi_action[2])
                self.state["passengers"][taxi_action[2]]["status"] = DELIVERED
            elif taxi_action[0] == "refuel":
                self.state["taxis"][taxi_action[1]]["fuel"] = self.state["taxis"][taxi_action[1]]["fuel_capacity"]
        return self.hash_state()

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        self.unhash_state(state)
        for passenger in self.passengers_names:
            if self.state["passengers"][passenger]["status"] != DELIVERED:
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate. """
        self.unhash_state(node.state)
        try:
            return self.h_on_steroids(node)
        except:
            self.unsolvable = True
            return self.h_1(node)

    def h_turns_approx(self, node):
        undelivered_passengers = (passenger for passenger in self.passengers_names
                                  if self.state["passengers"][passenger]["status"] != DELIVERED)

        turns_approximation = 0
        for passenger in undelivered_passengers:
            min_dist = float("inf")
            for taxi_name in self.taxi_names:
                taxi_to_passenger = manhattan(self.state["taxis"][taxi_name]["location"],
                                              self.state["passengers"][passenger]["location"])

                taxi_to_gas = gas_to_passenger = taxi_to_gas_to_passenger = float("inf")
                for g_loc in self.gas_stations:
                    temp1 = manhattan(self.state["taxis"][taxi_name]["location"], g_loc)
                    temp2 = manhattan(g_loc, self.state["passengers"][passenger]["location"])
                    if taxi_to_gas_to_passenger > temp1 + temp2:
                        taxi_to_gas = temp1
                        gas_to_passenger = temp2
                        taxi_to_gas_to_passenger = taxi_to_gas + gas_to_passenger

                passenger_to_destination = manhattan(self.state["passengers"][passenger]["location"],
                                                     self.state["passengers"][passenger]["destination"])

                passenger_to_gas = gas_to_destination = passenger_to_gas_to_destination = float("inf")
                for g_loc in self.gas_stations:
                    temp1 = manhattan(self.state["passengers"][passenger]["location"], g_loc)
                    temp2 = manhattan(g_loc, self.state["passengers"][passenger]["destination"])
                    if passenger_to_gas_to_destination > temp1 + temp2:
                        passenger_to_gas = temp1
                        gas_to_destination = temp2
                        passenger_to_gas_to_destination = passenger_to_gas + gas_to_destination

                if taxi_to_passenger > self.state["taxis"][taxi_name]["fuel"]:
                    taxi_to_passenger = taxi_to_gas_to_passenger
                elif taxi_to_passenger + passenger_to_destination > self.state["taxis"][taxi_name]["fuel"]:

                    if gas_to_passenger + passenger_to_destination <= self.state["taxis"][taxi_name]["fuel_capacity"]:
                        taxi_to_passenger = taxi_to_gas_to_passenger
                    else:
                        passenger_to_destination = passenger_to_gas_to_destination

                min_dist = min(min_dist, taxi_to_passenger + passenger_to_destination)

            turns_approximation = max(turns_approximation, min_dist)

        return turns_approximation

    def h_on_steroids(self, node):
        undelivered_passengers = (passenger for passenger in self.passengers_names
                                  if self.state["passengers"][passenger]["status"] != DELIVERED)

        turns_approximation = 0
        for passenger in undelivered_passengers:
            min_dist = float("inf")
            for taxi_name in self.taxi_names:
                taxi_to_passenger = self.graph_distance(self.state["taxis"][taxi_name]["location"],
                                                        self.state["passengers"][passenger]["location"])

                taxi_to_gas = gas_to_passenger = taxi_to_gas_to_passenger = float("inf")
                for g_loc in self.gas_stations:
                    temp1 = self.graph_distance(self.state["taxis"][taxi_name]["location"], g_loc)
                    temp2 = self.graph_distance(g_loc, self.state["passengers"][passenger]["location"])
                    if taxi_to_gas_to_passenger > temp1 + temp2:
                        taxi_to_gas = temp1
                        gas_to_passenger = temp2
                        taxi_to_gas_to_passenger = taxi_to_gas + gas_to_passenger

                passenger_to_destination = self.graph_distance(self.state["passengers"][passenger]["location"],
                                                               self.state["passengers"][passenger]["destination"])

                passenger_to_gas = gas_to_destination = passenger_to_gas_to_destination = float("inf")
                for g_loc in self.gas_stations:
                    temp1 = self.graph_distance(self.state["passengers"][passenger]["location"], g_loc)
                    temp2 = self.graph_distance(g_loc, self.state["passengers"][passenger]["destination"])
                    if passenger_to_gas_to_destination > temp1 + temp2:
                        passenger_to_gas = temp1
                        gas_to_destination = temp2
                        passenger_to_gas_to_destination = passenger_to_gas + gas_to_destination

                if taxi_to_passenger > self.state["taxis"][taxi_name]["fuel"]:
                    taxi_to_passenger = taxi_to_gas_to_passenger
                elif taxi_to_passenger + passenger_to_destination > self.state["taxis"][taxi_name]["fuel"]:

                    if gas_to_passenger + passenger_to_destination <= self.state["taxis"][taxi_name]["fuel_capacity"]:
                        taxi_to_passenger = taxi_to_gas_to_passenger
                    else:
                        passenger_to_destination = passenger_to_gas_to_destination

                min_dist = min(min_dist, taxi_to_passenger + passenger_to_destination)

            turns_approximation = max(turns_approximation, min_dist)

        return turns_approximation

    def h_on_steroids_no_gas(self, node):
        undelivered_passengers = (passenger for passenger in self.passengers_names
                                  if self.state["passengers"][passenger]["status"] != DELIVERED)

        turns_approximation = 0
        for passenger in undelivered_passengers:
            min_dist = float("inf")
            for taxi_name in self.taxi_names:
                taxi_to_passenger = self.graph_distance(self.state["taxis"][taxi_name]["location"],
                                                        self.state["passengers"][passenger]["location"])

                passenger_to_destination = self.graph_distance(self.state["passengers"][passenger]["location"],
                                                               self.state["passengers"][passenger]["destination"])

                min_dist = min(min_dist, taxi_to_passenger + passenger_to_destination)

            turns_approximation = max(turns_approximation, min_dist)

        return turns_approximation

    # def h_graph(self, node):
    #     counter = 0
    #     for passenger in self.passengers_names:
    #         counter += nx.astar_path_length(G=self.graph,
    #                                         source=self.state["passengers"][passenger]["location"],
    #                                         target=self.state["passengers"][passenger]["destination"])
    #     return counter / self.fleet_capacity

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        counter = 0
        for passenger in self.passengers_names:
            counter += 2 - self.state["passengers"][passenger]["status"]
        return counter / self.n_taxis

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        counter = 0
        for passenger in self.passengers_names:
            counter += math.fabs(self.state["passengers"][passenger]["location"][0] -
                                 self.state["passengers"][passenger]["destination"][0]) + \
                       math.fabs(self.state["passengers"][passenger]["location"][1] -
                                 self.state["passengers"][passenger]["destination"][1])
        return counter / self.n_taxis

    """ Feel free to add your own functions
    (-2, -2, None) means there was a timeout """


def create_taxi_problem(game):
    return TaxiProblem(game)
