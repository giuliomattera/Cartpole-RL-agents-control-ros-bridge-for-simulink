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
* model : store of all agents' models
* checkpoints : store of weights during training. Usefull for fine tuning of controller.

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
* To launch tensorboard:
```
tensorboard --logdir ./gradient_tape
```
* At the end of training you could use 'RLcontroller.py' agent with cart_pol.slx system.

## In progress...
* Developing of RLcontroller.py agent
