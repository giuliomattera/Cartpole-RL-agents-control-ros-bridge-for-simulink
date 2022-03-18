import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints/actor')
    os.makedirs('./checkpoints/critic')

if not os.path.exists('./model'):
    os.makedirs('./model')

#Agent configuration
NUM_STATES = 4
NUM_ACTIONS = 1
SCALE_EFFORT = 30
MAX_EFFORT = 30
STD = 0.2#0,1
STD_DECAY = 1.3
MU = 0 #-1,1
CRITIC_LR = 2e-3
ACTOR_LR = 1e-4
PANDA = False
TRAIN = True

#Simulation configuration
MAX_EPISODE = 100
TIME_STEP = 2e-3
