import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


if not os.path.exists('./model'):
    os.makedirs('./model')

#Simulation configuration
MAX_EPISODE = 100
TIME_STEP = 2e-3

#Agent configuration
NUM_STATES = 4
NUM_ACTIONS = 1

SCALE_EFFORT = 30
MAX_EFFORT = 30

STD = 1#0,1
MU = 0 #-1,1

EPSILON = 0.5
EPS_DECAY = 5/MAX_EPISODE

CRITIC_LR = 1e-3
ACTOR_LR = 2e-4

CLR_DECAY = 1/MAX_EPISODE
ALR_DECAY = 1/MAX_EPISODE

WARMUP = True
EPS_WARM = 20

EARLY_STOPPING = MAX_EPISODE
PANDA = True
TRAIN = True
BATCH_SIZE = 8

