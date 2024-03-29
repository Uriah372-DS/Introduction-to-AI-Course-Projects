additional_inputs = [
    {  # 6
        'optimal': False,
        "turns to go": 50,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P']],
        'taxis': {'taxi 1': {'location': (1, 3), 'fuel': 10, 'capacity': 3}},
        'passengers': {'Michael': {'location': (3, 4), 'destination': (2, 1),
                                   "possible_goals": ((2, 1), (3, 4)), "prob_change_goal": 0.1},
                       'Freyja': {'location': (0, 0), 'destination': (2, 1),
                                  "possible_goals": ((2, 1), (0, 0)), "prob_change_goal": 0.3}}
    },

    {  # 7
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P']],
        'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 8, 'capacity': 2}},
        'passengers': {'Eitan': {'location': (1, 0), 'destination': (0, 2),
                                 'possible_goals': ((0, 2), (1, 0)), 'prob_change_goal': 0.2},
                       'Omer': {'location': (3, 4), 'destination': (0, 2),
                                'possible_goals': ((0, 2), (1, 0)), 'prob_change_goal': 0.3}},
    },

    {  # 8
        'optimal': False,
        "turns to go": 50,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'P', 'G', 'P']],
        'taxis': {'taxi 1': {'location': (1, 2), 'fuel': 18, 'capacity': 1}},
        'passengers': {'Freyja': {'location': (2, 0), 'destination': (4, 2),
                                  'possible_goals': ((4, 2), (2, 0)), 'prob_change_goal': 0.2},
                       'Wolfgang': {'location': (2, 1), 'destination': (1, 4),
                                    'possible_goals': ((1, 4), (0, 0)), 'prob_change_goal': 0.8},
                       'Jacob': {'location': (3, 4), 'destination': (3, 2),
                                 'possible_goals': ((3, 2), (0, 0)), 'prob_change_goal': 0.2}},
    },

    {  # 9
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'P', 'G', 'P']],
        'taxis': {'taxi 1': {'location': (0, 2), 'fuel': 15, 'capacity': 2}},
        'passengers': {'Omar': {'location': (0, 3), 'destination': (3, 2),
                                'possible_goals': ((3, 2),), 'prob_change_goal': 0.8},
                       'John': {'location': (1, 0), 'destination': (4, 0),
                                'possible_goals': ((4, 0),), 'prob_change_goal': 0.8}},
    },

    {  # 10
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
        'taxis': {'taxi 1': {'location': (5, 6), 'fuel': 21, 'capacity': 3}},
        'passengers': {'Omer': {'location': (1, 5), 'destination': (2, 2),
                                'possible_goals': ((2, 2), (4, 3)), 'prob_change_goal': 0.2},
                       'Roee': {'location': (2, 1), 'destination': (4, 3),
                                'possible_goals': ((4, 3), (2, 2)), 'prob_change_goal': 0.5},
                       'Dana': {'location': (4, 2), 'destination': (5, 2),
                                'possible_goals': ((4, 3), (2, 2), (5, 2)), 'prob_change_goal': 0.2},
                       'Efrat': {'location': (5, 6), 'destination': (2, 3),
                                 'possible_goals': ((2, 3), (2, 2), (4, 3)), 'prob_change_goal': 0.3}},
    },

    {  # 11
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
        'taxis': {'taxi 1': {'location': (5, 5), 'fuel': 6, 'capacity': 2}},
        'passengers': {'Janet': {'location': (5, 4), 'destination': (1, 4),
                                 'possible_goals': ((1, 4), (5, 0), (3, 4)), 'prob_change_goal': 0.05},
                       'Omer': {'location': (1, 5), 'destination': (5, 0),
                                'possible_goals': ((1, 4), (5, 0), (3, 4)), 'prob_change_goal': 0.4},
                       'Oliver': {'location': (4, 4), 'destination': (3, 4),
                                  'possible_goals': ((1, 4), (5, 0), (3, 4)), 'prob_change_goal': 0.2}},
    },

    {  # 12
        'optimal': False,
        "turns to go": 50,
        'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
        'taxis': {'taxi 1': {'location': (1, 0), 'fuel': 6, 'capacity': 2}},
        'passengers': {'Yael': {'location': (5, 4), 'destination': (1, 6),
                                'possible_goals': ((1, 6), (3, 6), (4, 6)), 'prob_change_goal': 0.2},
                       'Janet': {'location': (5, 5), 'destination': (3, 6),
                                 'possible_goals': ((1, 6), (3, 6), (4, 6)), 'prob_change_goal': 0.5},
                       'Francois': {'location': (5, 0), 'destination': (4, 6),
                                    'possible_goals': ((1, 6), (3, 6), (4, 6)), 'prob_change_goal': 0.2}},
    },

    {  # 13
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P']],
        'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
                  'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
        'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4),
                                'possible_goals': ((1, 4),), 'prob_change_goal': 0.2},
                       'Daniel': {'location': (3, 1), 'destination': (2, 1),
                                  'possible_goals': ((2, 1), (0, 1), (3, 1)), 'prob_change_goal': 0.2},
                       'Freyja': {'location': (2, 3), 'destination': (2, 4),
                                  'possible_goals': ((2, 4), (3, 0), (3, 2)), 'prob_change_goal': 0.2},
                       'Tamar': {'location': (3, 0), 'destination': (3, 2),
                                 'possible_goals': ((3, 2),), 'prob_change_goal': 0.2}},
    },

    {  # 14
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'P', 'G', 'P']],
        'taxis': {'taxi 1': {'location': (4, 4), 'fuel': 17, 'capacity': 1},
                  'taxi 2': {'location': (2, 1), 'fuel': 18, 'capacity': 3}},
        'passengers': {'Freyja': {'location': (4, 2), 'destination': (4, 4),
                                  'possible_goals': ((4, 4), (4, 2), (0, 0)), 'prob_change_goal': 0.2},
                       'Dave': {'location': (0, 0), 'destination': (3, 3),
                                'possible_goals': ((3, 3), (4, 2), (0, 0)), 'prob_change_goal': 0.2},
                       'Janet': {'location': (1, 4), 'destination': (3, 3),
                                 'possible_goals': ((3, 3), (1, 4), (3, 1)), 'prob_change_goal': 0.2},
                       'Francois': {'location': (3, 1), 'destination': (0, 3),
                                    'possible_goals': ((0, 3), (4, 4), (2, 1)), 'prob_change_goal': 0.2}},
    },

    {  # 15
        'optimal': False,
        "turns to go": 200,
        'map': [['P', 'P', 'P', 'I', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P', 'P', 'I'],
                ['P', 'P', 'I', 'P', 'P', 'I', 'P'],
                ['P', 'G', 'P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'G', 'I', 'P', 'P', 'P']],
        'taxis': {'taxi 1': {'location': (4, 0), 'fuel': 20, 'capacity': 2},
                  'taxi 2': {'location': (4, 3), 'fuel': 10, 'capacity': 3}},
        'passengers': {'Dana': {'location': (2, 6), 'destination': (4, 2),
                                'possible_goals': ((4, 2), (3, 1), (2, 4)), 'prob_change_goal': 0.2},
                       'Uri': {'location': (3, 5), 'destination': (1, 4),
                               'possible_goals': ((1, 4), (1, 2), (1, 3)), 'prob_change_goal': 0.2},
                       'Ali': {'location': (1, 2), 'destination': (1, 3),
                               'possible_goals': ((1, 3), (3, 5), (1, 2)), 'prob_change_goal': 0.2},
                       'Daniel': {'location': (1, 3), 'destination': (3, 4),
                                  'possible_goals': ((3, 4),), 'prob_change_goal': 0.2},
                       'Wolfgang': {'location': (3, 1), 'destination': (3, 6),
                                    'possible_goals': ((3, 6), (2, 6), (3, 5)), 'prob_change_goal': 0.2},
                       'Noa': {'location': (2, 4), 'destination': (3, 6),
                               'possible_goals': ((3, 6), (4, 3), (2, 6), (3, 5)), 'prob_change_goal': 0.2},
                       'Ayelet': {'location': (1, 2), 'destination': (0, 4),
                                  'possible_goals': ((0, 4), (2, 4)), 'prob_change_goal': 0.2},
                       'Khaled': {'location': (5, 1), 'destination': (3, 6),
                                  'possible_goals': ((3, 6), (1, 2), (5, 1)), 'prob_change_goal': 0.2}},
    },
    {  # 16
        'optimal': True,
        "turns to go": 50,
        'map': [['P', 'P', 'G', 'P', 'P'], ],
        'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 10, 'capacity': 1}},
        'passengers': {'Michael': {'location': (0, 0), 'destination': (0, 4),
                                   "possible_goals": ((0, 0), (0, 4)), "prob_change_goal": 0.2},
                       }
    },
    {  # 17
        'optimal': True,
        "turns to go": 50,
        'map': [['P', 'P', 'P', 'P', 'P'], ],
        'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 40, 'capacity': 1}},
        'passengers': {'Michael': {'location': (0, 0), 'destination': (0, 4),
                                   "possible_goals": ((0, 0), (0, 4)), "prob_change_goal": 0.7},
                       }
    },
    {  # 18
        'optimal': True,
        "turns to go": 50,
        'map': [['P', 'P', 'P', 'P', 'G'], ],
        'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 4, 'capacity': 1}},
        'passengers': {'Michael': {'location': (0, 0), 'destination': (0, 0),
                                   "possible_goals": ((0, 0), (0, 4)), "prob_change_goal": 0.7},
                       'Din': {'location': (0, 4), 'destination': (0, 0),
                               "possible_goals": ((0, 0), (0, 4)), "prob_change_goal": 0.2},
                       }
    },
]
