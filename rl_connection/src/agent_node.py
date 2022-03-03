#!/usr/bin/env python3

import rospy
import tensorflow as tf
import numpy as np
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


class DDPG():

    def __init__(self, gamma=0.99, effort=1, mu=0, std=1, scale_effort = 1):
        super(DDPG, self).__init__()
        self.num_states = 1
        self.num_actions = 1
        self.gamma = gamma
        self.max_effort = effort
        self.mu = mu
        self.std = std
        self.scale_effort = scale_effort

    def Actor(self):

        input = tf.keras.layers.Input(shape=(self.num_states,))

        hidden = tf.keras.layers.Dense(30, activation="relu")(input)

        hidden2 = tf.keras.layers.Dense(10, activation="relu")(hidden)

        outputs = tf.keras.layers.Dense(
            self.num_actions, activation="tanh")(hidden)

        
        return tf.keras.Model(input, outputs)

    def Critic(self):

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        state_out_critic = tf.keras.layers.Dense(
            16, activation="relu")(state_input)
        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        action_out_critic = tf.keras.layers.Dense(
            16, activation="relu")(action_input)

        # Concatening 2 networks
        concat = tf.keras.layers.Concatenate()(
            [state_out_critic, action_out_critic])
        concat = tf.keras.layers.Dense(256, activation="relu")(concat)
        out = tf.keras.layers.Dense(128, activation="relu")(concat)

        # Predicted Q(s,a)
        outputs = tf.keras.layers.Dense(1)(out)

        return tf.keras.Model([state_input, action_input], outputs)

    def update(self, state, action, reward, next_state, next_action):
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        
        # Action is a float. Expand is needed
        action = tf.expand_dims(action, 0)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

        next_action = tf.expand_dims(next_action, 0)
        next_action = tf.convert_to_tensor(next_action, dtype=tf.float32)

        reward = tf.expand_dims(reward, 0)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as gradient:
            critic_q = critic_model([state, action], training=True)  # Q(s,a)
            y = reward + self.gamma * \
                critic_model([next_state, next_action],
                             training=True)  # Q(s',a')

            critic_loss = tf.math.reduce_mean(
                tf.math.square(y-critic_q))  # TD error

            critic_grad = gradient.gradient(
                critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(
                zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as gradient:
            action = actor_model(state, training=True)

            critic_q = critic_model([state, action], training=True)
            actor_loss = -tf.math.reduce_mean(critic_q)

            actor_grad = gradient.gradient(
                actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor_model.trainable_variables))

        return critic_loss, actor_loss

def send_action(pub, action_msg, action):
    rate = rospy.Rate(10)
    action_msg.data = action
    pub.publish(action_msg)
    rate.sleep()

def agent_out(agent, actor_model, state):

    s_nn = tf.expand_dims(state, 0) #prepare state for NN
    a = actor_model(s_nn)
    a = a + a*agents.std + agents.mu  # searching in action space
    a = a*agents.scale_effort
    a0 = np.clip(a.numpy(), -agents.max_effort, agents.max_effort)
    return a0

def state_callback(state):
    action = Float32()
    io = state.data[0]
    print('[INFO] Recvieved a state from simulink. ', io)
    a0 = agent_out(agents, actor_model, state.data)
    print('Taking an action ', a0, ' on sytem.')
    send_action(pub, action, a0)

def reward_callback(rew_sim):
    rew = rew_sim.data
    print('[INFO] Recieved a reward :' ,rew)


if __name__ == '__main__':
    try:
        ''' Initialiaze agent'''
        agents = DDPG()
        agents.num_states = 4
        agents.num_actions = 1
        agents.max_effort = 10
        agents.scale_effort = 1

        actor_model = agents.Actor()
        critic_model = agents.Critic()

        # Learning rate for actor-critic models
        critic_lr = 1e-5
        actor_lr = 1e-5
        agents.gamma = 0.99
        agents.mu = 0.1
        agents.std = 0.1

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        #Initialize agent node
        rospy.init_node('Agent')
        pub = rospy.Publisher('agent_action', Float32, queue_size=10)
        while rospy.is_shutdown() == False:
            rospy.Subscriber("state", Float32MultiArray, state_callback)
            rospy.Subscriber("reward", Float32, reward_callback)
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
