# Carte Pole control with DDPG agent

In this project a DDPG agent is created in python with tensorflow library. ROS is used to link the actions of the agent with the systems' state (Carte Pole) modelled in Simulink. 

## Technologies
Project is created with:
* MATLAB R2021a
* ROS Noetic
* Tensorflow 2.4

## Project sections
* simulink : you can found all simulink models and matlab node
* rl_connections : ROS module with the agent definition and nodes. You can found also a config file useful for agent_node
* bash file : for launching DDPG training with matlab brige. Note that is necessary to configurate the enviroment and agent!
* model : store of all agents' weights and model

## How launch
* Run roscore from terminal
```
roscore
```
* Run "agent_node.py" from new terminal:
```
rosrun rl_connection agent_node.py
```
* Run "matlab_node.m" from new terminal with launch file
* Alternatively
```
cd folder_prj
./launch.bash
```
* Use command to importTensorFlowNetwork from model folder

## In progress...
* Visualization of training on tensorboard
* Saving weights procedure
* Importing weights in simulink and test agent
