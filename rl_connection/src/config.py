import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


if not os.path.exists('./model'):
    os.makedirs('./model')

#Simulation configuration
MAX_EPISODE = 100
TIME_STEP = 1e-3

#Agent configuration
NUM_STATES = 4
NUM_ACTIONS = 1

SCALE_EFFORT = 30
MAX_EFFORT = 30

CRITIC_LR = 6e-4
ACTOR_LR = 3e-5

CLR_DECAY = 1/MAX_EPISODE
ALR_DECAY = 1/MAX_EPISODE

WARMUP = False
EPS_WARM = 5

PANDA = True
TRAIN = True
BATCH_SIZE = 8

