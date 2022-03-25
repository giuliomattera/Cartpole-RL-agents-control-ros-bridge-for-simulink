import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


if not os.path.exists('./model'):
    os.makedirs('./model')

#Simulation configuration
MAX_EPISODE = 50
TIME_STEP = 2e-2

#Agent configuration
NUM_STATES = 4
NUM_ACTIONS = 1

SCALE_EFFORT = 30
MAX_EFFORT = 30

STD = 1#0,1
MU = 0 #-1,1

EPSILON = 0.5
EPS_DECAY = 4/MAX_EPISODE

CRITIC_LR = 1e-3
ACTOR_LR = 3e-4

CLR_DECAY = 1/MAX_EPISODE
ALR_DECAY = 1/MAX_EPISODE

PANDA = True
TRAIN = True
BATCH_SIZE = 6

