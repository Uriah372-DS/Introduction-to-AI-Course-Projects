import random
from Simulator import Simulator

IDS = ['AI2']


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_taxis = []
        self.simulator = Simulator(initial_state)
        for taxi_name, taxi in initial_state['taxis'].items():
            if taxi['player'] == player_number:
                self.my_taxis.append(taxi_name)

    def act(self, state):
        actions = {}
        self.simulator.set_state(state)
        for taxi in self.my_taxis:
            actions[taxi] = set()
            neighboring_tiles = self.simulator.neighbors(state["taxis"][taxi]["location"])
            for tile in neighboring_tiles:
                actions[taxi].add(("move", taxi, tile))
            if state["taxis"][taxi]["capacity"] > 0:
                for passenger in state["passengers"].keys():
                    if state["passengers"][passenger]["location"] == state["taxis"][taxi]["location"]:
                        actions[taxi].add(("pick up", taxi, passenger))
            for passenger in state["passengers"].keys():
                if (state["passengers"][passenger]["destination"] == state["taxis"][taxi]["location"]
                        and state["passengers"][passenger]["location"] == taxi):
                    actions[taxi].add(("drop off", taxi, passenger))
            actions[taxi].add(("wait", taxi))

        while True:
            whole_action = []
            for atomic_actions in actions.values():
                for action in atomic_actions:
                    if action[0] == "drop off":
                        whole_action.append(action)
                        break
                    if action[0] == "pick up":
                        whole_action.append(action)
                        break
                else:
                    whole_action.append(random.choice(list(atomic_actions)))
            whole_action = tuple(whole_action)
            if self.simulator.check_if_action_legal(whole_action, self.player_number):
                return whole_action