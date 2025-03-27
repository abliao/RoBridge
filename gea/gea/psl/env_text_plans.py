ROBOSUITE_PLANS = {
    "Lift": [
        ("red cube", "grasp")
    ],
    "Door": [
        ("door", "grasp")
    ],
    "NutAssemblyRound": [
        ("silver round nut", "grasp"), 
        ("silver peg", "place")
    ],
    "NutAssemblySquare": [
        ("gold square nut", "grasp"), 
        ("gold peg", "place")
    ],
    "NutAssembly": [
        ("gold square nut", "grasp"), 
        ("gold peg", "place"), 
        ("silver round nut", "grasp"), 
        ("silver peg", "place")
    ],
    "PickPlaceCan": [
        ("red can", "grasp"), 
        ("bin4", "place")
    ], 
    "PickPlaceCereal": [
        ("cereal box", "grasp"), 
        ("bin3", "place")
    ],
    "PickPlaceBread": [
        ("bread", "grasp"), 
        ("bin2", "place")
    ],
    "PickPlaceMilk": [
        ("milk carton", "grasp"), 
        ("bin1", "place")
    ],
    "PickPlaceCerealMilk": [
        ("milk carton", "grasp"),
        ("bin1", "place"),
        ("cereal box", "grasp"),
        ("bin2", "place")
    ],
    "PickPlaceCanBread": [
        ("red can", "grasp"),
        ("bin2", "place"),
        ("bread", "grasp"),
        ("bin1", "place")
    ]
}

MOPA_PLANS = {
    "SawyerAssemblyObstacle-v0": [
        ("empty hole", "place")
    ],
    "SawyerLiftObstacle-v0": [
        ("red can", "grasp")
    ],
    "SawyerPushObstacle-v0": [
        ("red cube", "grasp")
    ]
}

METAWORLD_PLANS = {
    "assembly-v2": [
        ("green wrench", "grasp"), 
        ("small maroon peg", "place")
    ],
    "hammer-v2": [
        ("small red hammer handle", "grasp"), 
        ("gray nail on wooden box", "place")
    ],
    "bin-picking-v2": [
        ("small green cube", "grasp"), 
        ("blue bin", "place")
    ],
    "disassemble-v2":[
        ("green wrench handle", "grasp")
    ],
    "basketball-v2": [
        ("basketball", "grasp"), 
        ("basketballhoop", "place")
    ],
    "button-press-topdown-v2": [
        ("button", "press")
    ],
    "button-press-topdown-wall-v2": [
        ("button", "press")
    ],
    "button-press-v2": [
        ("button", "press")
    ],
    "button-press-wall-v2": [
        ("button", "press")
    ],
    "box-close-v2": [
        ("cover", "grasp"),
        ("box", "place"),
    ],
    "coffee-button-v2": [
        ("button", "press")
    ],
    "coffee-pull-v2": [
        ("mug", "pull")
    ],
    "coffee-push-v2": [
        ("mug", "push")
    ],
    "dial-turn-v2": [
        ("dial", "turn")
    ],
    "door-close-v2": [
        ("door", "close")
    ],
    "door-open-v2": [
        ("door", "open")
    ],
    "door-lock-v2": [
        ("doorlock", "turn")
    ],
    "door-unlock-v2": [
        ("doorlock", "turn")
    ],
    "hand-insert-v2": [
        ("block", "grasp"),
        ("hole", "place"),
    ],
    "drawer-close-v2": [
        ("drawer", "close")
    ],
    "drawer-open-v2": [
        ("drawer", "open")
    ],
    "faucet-close-v2": [
        ("faucet", "turn")
    ],
    "faucet-open-v2": [
        ("faucet", "turn")
    ],
    "handle-press-side-v2": [
        ("handle", "press")
    ],
    "handle-press-v2": [
        ("handle", "press")
    ],
    "handle-pull-side-v2": [
        ("handle", "turn")
    ],
    "handle-pull-v2": [
        ("handle", "turn")
    ],
    "lever-pull-v2": [
        ("lever", "turn")
    ],
    "peg-insert-side-v2": [
        ("peg", "grasp"),
        ("hole", "place"),
    ],
    "pick-place-v2": [
        ("cylinder", "grasp"),
        ("point", "place"),
    ],
    "pick-place-wall-v2": [
        ("cylinder", "grasp"),
        ("point", "place"),
    ],
    "pick-out-of-hole-v2": [
        ("cylinder", "grasp"),
        ("table", "place"),
    ],
    "reach-v2": [
        ("point", "reach")
    ],
    "push-back-v2": [
        ("cylinder", "pull")
    ],
    "push-v2": [
        ("cylinder", "push")
    ],
    "plate-slide-v2": [
        ("puck", "push")
    ],
    "plate-slide-side-v2": [
        ("puck", "push")
    ],
    "plate-slide-back-v2": [
        ("puck", "pull")
    ],
    "plate-slide-back-side-v2": [
        ("puck", "pull")
    ],
    "peg-unplug-side-v2": [
        ("peg", "grasp"),
        ("table", "place"),
    ],
    "soccer-v2": [
        ("soccer", "push"),
    ],
    "stick-push-v2": [
        ("stick", "grasp"),
        ("bottle", "push"),
    ],
    "stick-pull-v2": [
        ("stick", "grasp"),
        ("bottle", "pull"),
    ],
    "push-wall-v2": [
        ("cylinder", "grasp"),
        ("point", "push"),
    ],
    "reach-wall-v2": [
        ("point", "reach")
    ],
    "shelf-place-v2": [
        ("block", "grasp"),
        ("shelf", "place"),
    ],
    "sweep-into-v2": [
        ("block", "push")
    ],
    "sweep-v2": [
        ("block", "push")
    ],
    "window-open-v2": [
        ("window", "push")
    ],
    "window-close-v2": [
        ("window", "push")
    ],
}

KITCHEN_PLANS = {
    "kitchen-microwave-v0": [
        ("microwave handle", "grasp")
    ],
    "kitchen-slide-v0": [
        ("slide", "grasp")
    ],
    "kitchen-kettle-v0": [
        ("kettle", "grasp")
    ],
    "kitchen-light-v0": [
        ("light", "grasp")
    ],
    "kitchen-tlb-v0": [
        ("top burner", "grasp")
    ],
    "kitchen-ms5-v0": [
        ("microwave handle", "grasp"),
        ("kettle", "grasp"),
        ("light", "grasp"),
        ("top burner", "grasp"),
        ("slide", "grasp"),
    ],
    "kitchen-ms6-v0": [
        ("microwave handle", "grasp"),
        ("kettle", "grasp"),
        ("light", "grasp"),
        ("top burner", "grasp"),
        ("slide", "grasp"),
        ("bottom right burner", "grasp"),
    ],
    "kitchen-ms7-v0": [
        ('kettle', 'grasp'), 
        ('light switch', 'grasp'), 
        ('slide cabinet', 'grasp'), 
        ('top burner', 'grasp'), 
        ('microwave', 'grasp'), 
        ('bottom right burner', 'grasp'), 
        ('hinge cabinet', 'grasp'), 
    ],
    "kitchen-ms8-v0": [
        ("microwave handle", "grasp"),
        ("kettle", "grasp"),
        ("light", "grasp"),
        ("top burner", "grasp"),
        ("slide", "grasp"),
        ("bottom right burner", "grasp"),
        ("hinge cabinet", "grasp"),
        ("bottom left burner", "grasp"),
    ],
    "kitchen-ms10-v0": [
        ("microwave", "grasp"),
        ("kettle", "grasp"),
        ("close microwave", "grasp"),
        ("slide", "grasp"),
        ("top burner", "grasp"),
        ("close slide", "grasp"),
        ("hinge cabinet", "grasp"),
        ("light", "grasp"),
        ("close hinge cabinet", "grasp"),
        ("bottom right burner", "grasp"),
    ],
    "kitchen-ms10-v2": [
        ('kettle', 'grasp'), 
        ('light switch', 'grasp'), 
        ('slide cabinet', 'grasp'), 
        ('top burner', 'grasp'), 
        ('bottom right burner', 'grasp'), 
        ('top right burner', 'grasp'), 
        ('bottom left burner', 'grasp'), 
        ('microwave', 'grasp'), 
        ('hinge cabinet', 'grasp'), 
        ('close microwave', 'grasp')
    ],
    "kitchen-ms10-v1": [
        ('kettle', 'grasp'), 
        ('light switch', 'grasp'), 
        ('slide cabinet', 'grasp'), 
        ('top burner', 'grasp'), 
        ('microwave', 'grasp'), 
        ('bottom right burner', 'grasp'), 
        ('hinge cabinet', 'grasp'), 
        ('top right burner', 'grasp'), 
        ('bottom left burner', 'grasp'), 
        ('close microwave', 'grasp')
    ],
    "kitchen-ms3-v0": [
        ("kettle", "grasp"),
        ("light switch", "grasp"),
        ("top burner", "grasp"),
    ],
    "kitchen-kettle-burner-v0": [
        ("kettle", "grasp"),
        ("top burner", "grasp"),
    ]
}
