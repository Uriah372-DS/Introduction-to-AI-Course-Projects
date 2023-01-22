from copy import deepcopy
import logging
import random

# DIMENSIONS = (10, 10)
PASSENGER_ARRIVAL_PROBABILITY = 0.2
PASSENGER_NAMES = ["Yossi", "Yael", "Dana", "Kobi", "Avi", "Noa", "John", "Dave", "Mohammad", "Sergei", "Nour", "Ali",
                   "Janet", "Francois", "Greta", "Freyja", "Jacob", "Emma", "Meytal", "Oliver", "Roee", "Omer", "Omar",
                   "Reema", "Gal", "Wolfgang", "Michael", "Efrat", "Iris", "Eitan", "Amir", "Khaled", "Jana", "Moshe",
                   "Lian", "Irina", "Tamar", "Ayelet", "Uri", "Daniel"]


class Simulator:
    def __init__(self, initial_state):
        self.state = deepcopy(initial_state)
        self.score = {'player 1': 0, 'player 2': 0}
        self.dimensions = len(self.state['map']), len(self.state['map'][0])
        self.turns_to_go = self.state['turns to go']

    def neighbors(self, location):
        """
        return the neighbors of a location
        """
        x, y = location[0], location[1]
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for neighbor in tuple(neighbors):
            if neighbor[0] < 0 or neighbor[0] >= self.dimensions[1] or neighbor[1] < 0 or neighbor[1] >= \
                    self.dimensions[0] or self.state['map'][neighbor[0]][neighbor[1]] != 'P':
                neighbors.remove(neighbor)
        return neighbors

    def check_if_action_legal(self, action, player):
        def _is_move_action_legal(move_action, player):
            taxi_name = move_action[1]
            if taxi_name not in self.state['taxis'].keys():
                logging.error(f"Taxi {taxi_name} does not exist!")
                return False
            if player != self.state['taxis'][taxi_name]['player']:
                logging.error(f"Taxi {taxi_name} does not belong to player {player}!")
                return False
            l1 = self.state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            if l2 not in self.neighbors(l1):
                logging.error(f"Taxi {taxi_name} cannot move from {l1} to {l2}!")
                return False
            return True

        def _is_pick_up_action_legal(pick_up_action, player):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if player != self.state['taxis'][taxi_name]['player']:
                return False
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if self.state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            return True

        def _is_drop_action_legal(drop_action, player):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if player != self.state['taxis'][taxi_name]['player']:
                return False
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['destination']:
                return False
            # check passenger is in the taxi
            if self.state['passengers'][passenger_name]['location'] != taxi_name:
                return False
            return True

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

        players_taxis = [taxi for taxi in self.state['taxis'].keys() if self.state['taxis'][taxi]['player'] == player]

        if len(action) != len(players_taxis):
            logging.error(f"You had given {len(action)} atomic commands, while you control {len(players_taxis)}!")
            return False
        for atomic_action in action:
            # trying to act with a taxi that is not yours
            if atomic_action[1] not in players_taxis:
                logging.error(f"Taxi {atomic_action[1]} is not yours!")
                return False
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action, player):
                    logging.error(f"Move action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action, player):
                    logging.error(f"Pick action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action, player):
                    logging.error(f"Drop action {atomic_action} is illegal!")
                    return False
            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            # logging.error(f"Actions {action} are mutex!")
            return False
        # check taxis collision
        if len(self.state['taxis']) > 1:
            taxis_location_dict = dict(
                [(t, self.state['taxis'][t]['location']) for t in self.state['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                # logging.error(f"Actions {action} cause collision!")
                return False
        return True

    def apply_action(self, action, player):
        for atomic_action in action:
            self._apply_atomic_action(atomic_action, player)
        self.turns_to_go -= 1

    def _apply_atomic_action(self, atomic_action, player):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            self.state['taxis'][taxi_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            self.state['taxis'][taxi_name]['capacity'] -= 1
            self.state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            self.state['taxis'][taxi_name]['capacity'] += 1
            self.score[f"player {player}"] += self.state['passengers'][passenger_name]['reward']
            del self.state['passengers'][passenger_name]
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def add_passenger(self):
        if len(self.state['passengers']) > 25:
            return
        if random.random() < PASSENGER_ARRIVAL_PROBABILITY:
            while True:
                passenger_name = random.choice(PASSENGER_NAMES)
                if passenger_name not in self.state['passengers'].keys():
                    break
            while True:
                passenger_location = (
                random.randint(0, self.dimensions[0] - 1), random.randint(0, self.dimensions[1] - 1))
                if self.state['map'][passenger_location[0]][passenger_location[1]] == 'P':
                    break
            while True:
                passenger_destination = (
                random.randint(0, self.dimensions[0] - 1), random.randint(0, self.dimensions[1] - 1))
                if self.state['map'][passenger_location[0]][passenger_location[1]] == 'P':
                    break
            reward = random.randint(1, 9)
            self.state['passengers'][passenger_name] = {'location': passenger_location,
                                                        'destination': passenger_destination,
                                                        'reward': reward}

    def act(self, action, player):
        if self.check_if_action_legal(action, player):
            self.apply_action(action, player)
            self.add_passenger()
        else:
            raise ValueError(f"Illegal action!")

    def print_scores(self):
        print(f"Scores: player 1: {self.score[0]}, player 2: {self.score[1]}")

    def print_state(self):
        for key, value in self.state.items():
            print(f'{key}:')
            try:
                for secondary_key, secondary_value in value.items():
                    print(f"{secondary_key}: {secondary_value}")
            except AttributeError:
                if key == 'map':
                    for row in value:
                        print(row)
                else:
                    print(f"{self.turns_to_go}")
            print("------------------")

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_score(self):
        return self.score
