import os

if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')


if not os.path.exists('./model'):
    os.makedirs('./model')

#Simulation configuration
MAX_EPISODE = 1000
TS = 1e-3

CLR_DECAY =1//MAX_EPISODE
ALR_DECAY = 1//MAX_EPISODE

# Hyper-parameters
WARMUP = False
EPS_WARM = 10

#Learning strategies
PANDA = False
TRAIN = True

TYPE = 'QAC'

