# Carte Pole control with RL agents

In this project a REINFORCE with baseline (Actor-Critic) and DDPG agents are created in python with tensorflow library. ROS is used to link the actions of one of the agents with the systems' state (Carte Pole) modelled in Simulink (using Multibody Simscape library). OpenAI gym is used like benchmark to easily discover bugs in agent's code.

<details>
<summary><strong>Remarks on : SARSA agent</strong></summary>
  Q-Learning technique is an Off Policy technique beacuse uses the greedy approach to learn the Q-value. SARSA technique, on the other hand, is an On Policy and uses the action performed by the current policy to learn the Q-value:
  
![immagine](https://user-images.githubusercontent.com/97847032/162726373-a6c2e706-7e56-4cc6-8c3c-132538810160.png)
  
  To train the network a gradient descent method is used to reduce the temporal difference error.
  
</details>
  
<details>
<summary><strong>Remarks on : Actor-Critic agent with baseline for continous action space</strong></summary>
  
Unlike many other RL algorithms that parameterize the value functions (Q learning, SARSA, DQN etc.) and derive the policy from the optimal value function using off-policy or on-policy methods (check in resources for details), the **Policy gradient algorithms** use a neural network or a function (to be more general) to estimate directly the policy. To do this the core idea is to maximize the V function, so for use a **gradient ascent** this function must be differentiable, this means that the policy will be a softmax, a gaussian distribution or a neural network (it depends if action space is discrete or continous). REINFORCE is a popular algortihms (check resources for more details) that use the **temporal difference error** coming from Bellamn equation to calculate the gradient:
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161729301-381c7cdd-380e-44ba-b2a8-96608dc95b01.png)

Where Gt is the comulative discounted reward at each time step, The learning algortihms look like:
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161730201-49d4261c-0836-4496-87c8-6d5fbd618a5b.png)

But also if the policy gradients methods could be simple to implement, the major backside is the high variance caused by the calculation of returns (or reward). A common way to reduce variance is subtract a **baseline b(s)** from the returns in the policy gradient that does not depend from the action taken from policy in this way it mustn’t introduce any bias to the policy gradient. This could be also a random number, but a great candidate to use like baseline is the the value function itself! 
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161731662-5c5e3308-63fb-4ad6-a85e-b05e61c59462.png)

 
 This bring to a new kind of agent called **actor-critic**, in which the actor is the part of the agent that take the action, and the critic is the part that evaluate how the state (or the pair of (action,state)) is good. If we use as baseline the value Q(s,a) we obtain a Q-Actor-Critic agent. The goal is to minimize the TD error for the critic and use PG algorithm for the actor, like shown before. 
  
  If we are talking about Deep Reinforcement Learning, we have to train 2 differents networks:
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161732104-cdcd655b-66b8-4183-b8d9-3f0071b46d50.png)
  
  Now if the action space is discrete, we could use a softmax activation function in the last layer of actor network, otherwise we have to split our actor network in 2 output with dimension equal to n (number of actions) one for predict the mean of a gaussian distribution and another to predict the ln(std) (beacuse it allow us to predict any value) and we will use these 2 parameters to sample an action according with this distribution (this for all sample in the mu-vector and ln(std)-vector output). This is the different between a **categorical policy** and **fiagonal gaussian policy** for a **stochastic policy**

</details>


<details>
<summary><strong>Remarks on : Deep Deterministic Policy Gradient agent</strong></summary>
In this case we have 4 networks:
  
* Actor network that predict direcrtly the action istead predict [mu,std] of a gaussian distribution
* Target Actor
* Q network, quite similar to baseline network in AC agent
* Target Q network
 
  The target networks are time-delayed copies of their original networks that we use to upgrade weights in quite stable way. In methods that do not use target networks, the update equations of the network are interdependent on the values calculated by the network itself, which makes it prone to divergence.
  
* DDPG is an off-policy algorithm, because is a sort of extension of Q-learning.
* DDPG can only be used for environments with continuous action spaces.
* DDPG can be thought of as being deep Q-learning for continuous action spaces.
  
 Recap the Bellman equation describing the optimal action-value function in Q-learning algorithm (like you can see is off-policy):
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161815819-fda3b998-3262-4dd7-9272-a77af8217b61.png)

  If Q(s,a) function is approximated by a neural network we could use a mean-squared Bellman error (MSBE) function to train it (like for the critic of QAC) :
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161816035-5129143c-64e2-4207-b802-783d59b3ba45.png)

 At this point we have to add 2 stuff respect to Q-learning:
  * A buffer to store all trajectory
  * Q(s',a') is called **target network**
  
  In DQN-based algorithms, the target network is just copied over from the main network every some-fixed-number of steps. In DDPG-style algorithms, the target network is updated once per main network update by polyak averaging. 
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161816839-c2e25d02-22a1-4852-a37d-2e01bfd25e99.png)

  Policy learning in DDPG is fairly simple. We want to learn a deterministic policy which gives the action that maximizes Q(s,a). Because the action space is continuous, and we assume the Q-function is differentiable with respect to action, we can just perform gradient ascent (with respect to policy parameters only) to solve
  
![immagine](https://user-images.githubusercontent.com/97847032/161816871-630ae12e-4005-44fb-b003-342a10df0535.png)
 
 Once DDPG is a off-policy algortihms and the policy is **deterministic** (not stochastic like REINFORCE with baseline), if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add white noise to their actions at training time. In the DDPG paper (see references) an Ornstein-Uhlenbeck process is used.
  
 ![immagine](https://user-images.githubusercontent.com/97847032/161817384-95e84e9e-035b-420d-9ffc-360db622cc6b.png)

  ![immagine](https://user-images.githubusercontent.com/97847032/161817856-c267ee10-c504-48a2-9663-cb6657ca038d.png)
  
  So, for continuous action signals, it is important to set the noise standard deviation appropriately to encourage exploration. An hint : If your agent converges on local optima too quickly, promote agent exploration by increasing the amount of noise.

</details>

<details>
<summary><strong>Remarks on : Robot Operating System and ROS Toolbox</strong></summary>
ROS stands for Robot Operating System. Even if it says so, ROS is not a real operating system since it goes on top of Linux Ubuntu. ROS is a framework on top of the O.S. that allows it to abstract the hardware from the software. This means you can think in terms of software for all the hardware of the robot. ROS has a communication protocol, principally based on publisher/subscriber (but not only), that allow you to send a message from your software module (one node) to robot controller (another node) using special messages and topics.
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161818837-d60a50a4-85f1-499c-b785-45024f89cfe6.png)
  
  ROS Toolbox allow you to exchange messages from differents nodes deployed everywhere (on a HPC SBC, IPC,..) with MATLAB ecosystem. 
  
  ![immagine](https://user-images.githubusercontent.com/97847032/161819124-17971d79-9402-4597-88c5-6f5f6e90347e.png)

  Once a node is created with python ROS API (rospy) and another one is created with MATLAB (in the same computer, but also in other one) these can communicate using standard messages once topics are defined. In the resource you will find a book and some courses to getting started with ROS.

 </details>

## Technologies
Project is created with:
* MATLAB R2021a : ROS Toolbox for MATLAB and Simulink. Simscape Multibody library for kinematic and dynamic model of system
* ROS Noetic : std messages (Float32, Float32MultiArray, Bool) and Pub/Subscriber communication
* Tensorflow 2.4.0 : keras API and gradient tape method for learning
* TensorFlow Probability 0.12.2
* OpenAI gym

## Project sections
* simulink : you can found all simulink models and matlab node
* rl_connections : ROS module with the agent definition and nodes. You can found also a config file useful for agent_node
* bash file : for launching RL agent training with matlab brige. Note that is necessary to configurate the enviroment and agent!
* model : store of all agents' models
* checkpoints : store of weights during training. Usefull for fine tuning of controller.
* benchmarks : you can found some jupyter notebooks with test of agents hyperarameters in gym (pendulum env)

## How launch
Once ROS Noetic full desktop version is installed and your catkin_ws is created, you can download all files and put they in your src folder. 

* Run roscore from terminal to initilize the ROS master node
```
roscore
```
* Run "DDPG.py" or "AC.py" from new terminal (rl_connection is the name of folder in our src):
```
rosrun rl_connection DDPG.py
```
* Run "matlab_node.m" from new terminal with launch file or from MATLAB GUI with run button.
* Alternatively
```
cd folder_prj
./launch.bash
```
* To launch tensorboard (be sure that you are in your catkin_ws folder):
```
tensorboard --logdir ./gradient_tape
```
* At the end of training you could use set Training == False in config file to use agent node (DDPG or QAC) as controller

## Resoruces

* [Getting started with key concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
* [On-policy vs off-policy alghoritms](https://analyticsindiamag.com/reinforcement-learning-policy/)
* REINFORCE with baseline (AC) from Sutton and Barto, chapter 13
* [DPPG paper](https://arxiv.org/pdf/1509.02971.pdf)
* [DPPG implementation](https://keras.io/examples/rl/ddpg_pendulum/4)
* [What is ROS?](https://www.theconstructsim.com/what-is-ros/)
* [ROS Toolbox](https://it.mathworks.com/products/ros.html)
* [Mastering ROS, Cacace, Lentin](https://www.amazon.com/Mastering-ROS-Robotics-Programming-troubleshooting/dp/1801071020)
* [ROS for beginners by Anis Koubaa](https://www.udemy.com/course/ros-essentials/)
* [What is gym?](https://gym.openai.com/)

<details>
<summary><strong>Video examples </strong></summary>
  In progress...

</details>




