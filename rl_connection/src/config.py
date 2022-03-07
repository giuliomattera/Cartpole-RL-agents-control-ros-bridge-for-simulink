import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints/actor')
    os.makedirs('./checkpoints/critic')
if not os.path.exists('./model'):
    os.makedirs('./model')
#Agent configuration
NUM_STATES = 1
NUM_ACTIONS = 1
SCALE_EFFORT = 10
MAX_EFFORT = 10
CRITIC_LR = 1e-5
ACTOR_LR = 1e-5
PANDA = False

#Simulation configuration
MAX_EPISODE = 5
TIME_STEP = 1e-2
