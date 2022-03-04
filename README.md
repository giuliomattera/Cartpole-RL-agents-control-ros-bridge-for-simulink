# Carte Pole control with DDPG agent

In this project a DDPG agent is created in python with tensorflow library. ROS is used to link the actions of the agent with the systems' state (Carte Pole) modelled in Simulink. 

## Technologies
Project is created with:
* MATLAB R2021a
* ROS Noetic
* Tensorflow 2.4

## Project sections
* Simulink folder : you can found all simulink models
* rl_connections : ROS module with the agent definition and nodes

## How launch

* Run "matlab_node.m"
* Run "agent_node.py" from terminal:
```
roscore
rosrun rl_connection agent_node.py
```
* Use command to importTensorFlowNetwork from model folder
