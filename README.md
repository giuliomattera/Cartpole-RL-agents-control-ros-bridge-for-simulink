# Carte Pole control with RL agents

In this project a REINFORCE with baseline (Q-Actor-Critic) and DPPG agents are created in python with tensorflow library. ROS is used to link the actions of one of the agent with the systems' state (Carte Pole) modelled in Simulink. 

## Technologies
Project is created with:
* MATLAB R2021a : ROS Toolbox for MATLAB and Simulink. Simscape Multibody library for kinematic and dynamic model of system
* ROS Noetic : std messages (Float32, Float32MultiArray, Bool) and Pub/Subscriber communication
* Tensorflow 2.4.0 : keras API and gradient tape method for learning
* TensorFlow Probability 0.12.2

## Project sections
* simulink : you can found all simulink models and matlab node
* rl_connections : ROS module with the agent definition and nodes. You can found also a config file useful for agent_node
* bash file : for launching RL agent training with matlab brige. Note that is necessary to configurate the enviroment and agent!
* model : store of all agents' models
* checkpoints : store of weights during training. Usefull for fine tuning of controller.

## How launch
* Run roscore from terminal
```
roscore
```
* Run "DDPG.py" or "QAC.p" from new terminal:
```
rosrun rl_connection DDPG.py
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
* At the end of training you could use set Training == False in config file to use agent_node as controller

##`Resoruces

* DPPG implementation : https://keras.io/examples/rl/ddpg_pendulum/
* DPPG paper : https://arxiv.org/pdf/1509.02971.pdf
* REINFORCE with baseline (QAC) from Sutton and Barto, chapter 13 (in progress...)

## Example

https://user-images.githubusercontent.com/97847032/158992913-9c8decd3-f4b2-4580-b80f-9d37c355f094.mp4



https://user-images.githubusercontent.com/97847032/158999496-45d29e59-8d7f-4827-a172-86489c18e255.mp4




