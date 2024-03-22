IDS = ["Your IDS here"]

class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS

    def act(self, state):
        raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS

    def selection(self, UCT_tree):
        raise NotImplementedError

    def expansion(self, UCT_tree, parent_node):
        raise NotImplementedError

    def simulation(self):
        raise NotImplementedError

    def backpropagation(self, simulation_result):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError
