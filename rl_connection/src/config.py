import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


if not os.path.exists('./model'):
    os.makedirs('./model')

#Simulation configuration
MAX_EPISODE = 500
TS = 1e-3

CLR_DECAY = 0
ALR_DECAY = 0

# Hyper-parameters
WARMUP = False
EPS_WARM = 5

#Learning strategies
PANDA = True
TRAIN = False
