def get_par(env_name):
    tc = 0
    max_eva = 0
    if env_name == 'Walker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BridgeWalker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BidirectionalWalker-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'Carrier-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Carrier-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Pusher-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Pusher-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Thrower-v0':
        tc = 300
        max_eva = 150
    elif env_name == 'Catcher-v0':
        tc = 400
        max_eva = 200
    elif env_name == 'BeamToppler-v0':
        tc = 1000
        max_eva = 100
    elif env_name == 'BeamSlider-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Lifter-v0':
        tc = 300
        max_eva = 200
    elif env_name == 'Climber-v0':
        tc = 400
        max_eva = 150
    elif env_name == 'Climber-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Climber-v2':
        tc = 1000
        max_eva = 200
    elif env_name == 'UpStepper-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'DownStepper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'ObstacleTraverser-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'ObstacleTraverser-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Hurdler-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'PlatformJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'GapJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Traverser-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'CaveCrawler-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'AreaMaximizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'AreaMinimizer-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'WingspanMazimizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'HeightMaximizer-v0':
        tc = 500
        max_eva = 150
    elif env_name == 'Flipper-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Jumper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Balancer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Balancer-v1':
        tc = 600
        max_eva = 150

    return max_eva, tc
