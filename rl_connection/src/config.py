import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints/actor')
    os.makedirs('./checkpoints/critic')
if not os.path.exists('./model'):
    os.makedirs('./model')
#Agent configuration
NUM_STATES = 4
NUM_ACTIONS = 1
SCALE_EFFORT = 25
MAX_EFFORT = 24
STD = 0.0 #0,1
MU = 0.0 #-1,1
CRITIC_LR = 1e-2
ACTOR_LR = 1e-2
PANDA = True
BATCH = 16

#Simulation configuration
MAX_EPISODE = 15
TIME_STEP = 1e-3
