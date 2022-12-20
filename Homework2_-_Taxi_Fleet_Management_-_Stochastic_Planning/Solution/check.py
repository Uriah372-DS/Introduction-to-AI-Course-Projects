import random
import networkx as nx

from ex2 import TaxiAgent, ids, OptimalTaxiAgent
from additional_inputs import additional_inputs
from inputs import small_inputs
import logging
import time
from copy import deepcopy

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1


def initiate_agent(state):
    """
    initiate the agent with the given state
    """
    if state['optimal']:
        return OptimalTaxiAgent(state)
    return TaxiAgent(state)


class EndOfGame(Exception):
    """
    Exception to be raised when the game is over
    """
    pass


class TaxiStochasticProblem:

    def __init__(self, an_input):
        """
        initiate the problem with the given input
        """
        self.initial_state = deepcopy(an_input)
        self.state = deepcopy(an_input)
        self.graph = self.build_graph()
        start = time.perf_counter()
        self.agent = initiate_agent(self.state)
        end = time.perf_counter()
        if end - start > INIT_TIME_LIMIT:
            logging.critical("timed out on constructor")
            raise TimeoutError
        self.score = 0

    def run_round(self):
        """
        run a round of the game
        """
        while self.state["turns to go"]:
            start = time.perf_counter()
            action = self.agent.act(self.state)
            end = time.perf_counter()
            if end - start > TURN_TIME_LIMIT:
                logging.critical(f"timed out on an action")
                raise TimeoutError
            if not self.is_action_legal(action):
                logging.critical(f"You returned an illegal action!")
                raise RuntimeError
            self.result(action)
        self.terminate_execution()

    def is_action_legal(self, action):
        """
        check if the action is legal
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
            if self.state['passengers'][passenger_name]['destination'] == self.state['passengers'][passenger_name]['location']:
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
            if self.state['map'][i][j] == 'G':
                return True
            else:
                return False

        def _is_action_mutex(global_action):
            assert type(global_action) == tuple, "global action must be a tuple"
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
            logging.error(f"You had given {len(action)} atomic commands, while there are {len(self.state['taxis'])}"
                          f" taxis in the problem!")
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    logging.error(f"Move action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action):
                    logging.error(f"Pick action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action):
                    logging.error(f"Drop action {atomic_action} is illegal!")
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action):
                    logging.error(f"Refuel action {atomic_action} is illegal!")
                    return False
            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            logging.error(f"Actions {action} are mutex!")
            return False
        # check taxis collision
        if len(self.state['taxis']) > 1:
            taxis_location_dict = dict([(t, self.state['taxis'][t]['location']) for t in self.state['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                logging.error(f"Actions {action} cause collision!")
                return False
        return True

    def result(self, action):
        """"
        update the state according to the action
        """
        self.apply(action)
        if action != "reset":
            self.environment_step()

    def apply(self, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment()
            return
        if action == "terminate":
            self.terminate_execution()
        for atomic_action in action:
            self.apply_atomic_action(atomic_action)

    def apply_atomic_action(self, atomic_action):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            self.state['taxis'][taxi_name]['location'] = atomic_action[2]
            self.state['taxis'][taxi_name]['fuel'] -= 1
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            self.state['taxis'][taxi_name]['capacity'] -= 1
            self.state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            self.state['passengers'][passenger_name]['location'] = self.state['taxis'][taxi_name]['location']
            self.state['taxis'][taxi_name]['capacity'] += 1
            self.score += DROP_IN_DESTINATION_REWARD
            return
        elif atomic_action[0] == 'refuel':
            self.state['taxis'][taxi_name]['fuel'] = self.initial_state['taxis'][taxi_name]['fuel']
            self.score -= REFUEL_PENALTY
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def environment_step(self):
        """
        update the state of environment randomly
        """
        for p in self.state['passengers']:
            passenger_stats = self.state['passengers'][p]
            if random.random() < passenger_stats['prob_change_goal']:
                # change destination
                passenger_stats['destination'] = random.choice(passenger_stats['possible_goals'])
        self.state["turns to go"] -= 1
        return

    def reset_environment(self):
        """
        reset the state of the environment
        """
        self.state["taxis"] = deepcopy(self.initial_state["taxis"])
        self.state["passengers"] = deepcopy(self.initial_state["passengers"])
        self.state["turns to go"] -= 1
        self.score -= RESET_PENALTY
        return

    def terminate_execution(self):
        """
        terminate the execution of the problem
        """
        print(f"End of game, your score is {self.score}!")
        print(f"-----------------------------------")
        raise EndOfGame

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial_state['map']), len(self.initial_state['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial_state['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

def main():
    """
    main function
    """
    print(f"IDS: {ids}")
    for an_input in small_inputs:
        try:
            my_problem = TaxiStochasticProblem(an_input)
            my_problem.run_round()
        except EndOfGame:
            continue
    for an_input in additional_inputs:
        try:
            my_problem = TaxiStochasticProblem(an_input)
            my_problem.run_round()
        except EndOfGame:
            continue


if __name__ == '__main__':
    main()
