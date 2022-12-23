small_inputs = [
    # 3x3 1 taxi, 1 passenger w/ 2 possible goals, plenty of fuel
    {
        "optimal": True,
        "map": [['P', 'P', 'P'],
                ['P', 'G', 'P'],
                ['P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}},
        "turns to go": 100
    },
    # 3x3 1 taxi, 1 passenger w/ 2 possible goals, low fuel
    {
        "optimal": True,
        "map": [['P', 'P', 'P'],
                ['P', 'G', 'P'],
                ['P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 3, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}},
        "turns to go": 100
    },
    # 3x3 2 taxi, 2 passengers w/ 1 possible goals, low fuel
    {
        "optimal": False,
        "map": [['P', 'P', 'G'],
                ['P', 'P', 'P'],
                ['G', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 3, "capacity": 1},
                  'taxi 2': {"location": (0, 1), "fuel": 3, "capacity": 1}},
        "passengers": {'Dana': {"location": (0, 2), "destination": (2, 2),
                                "possible_goals": ((2, 2),), "prob_change_goal": 0.1},
                       'Dan': {"location": (2, 0), "destination": (2, 2),
                               "possible_goals": ((2, 2),), "prob_change_goal": 0.1}
                       },
        "turns to go": 100
    },
    # 3x4 1 taxi, 1 passenger w/ 2 possible goals, low fuel
    {
        "optimal": False,
        "map": [['P', 'P', 'P', 'P'],
                ['I', 'I', 'I', 'G'],
                ['P', 'P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 8, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 0), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 0)), "prob_change_goal": 0.01}},
        "turns to go": 100
    },
    # 4x4 2 taxi, 1 passenger w/ 2 possible goals, no gas station
    {
        "optimal": False,
        "map": [['P', 'P', 'P', 'P'],
                ['I', 'I', 'I', 'P'],
                ['I', 'I', 'I', 'P'],
                ['P', 'P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 12, "capacity": 1},
                  'taxi 2': {"location": (3, 0), "fuel": 12, "capacity": 1}},
        "passengers": {'Dana': {"location": (3, 0), "destination": (2, 3),
                                "possible_goals": ((2, 3), (3, 2)), "prob_change_goal": 0.5}},
        "turns to go": 100
    }
]


