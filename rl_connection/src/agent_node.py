#!/usr/bin/env python3
import rospy
import tensorflow as tf
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, Bool
import datetime
import config

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
        #state = tf.expand_dims(state, 0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        #next_state = tf.expand_dims(next_state, 0)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        #action = tf.expand_dims(action, 0)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

        #next_action = tf.expand_dims(np.float32(next_action), 0)
        next_action = tf.convert_to_tensor(next_action, dtype=tf.float32)

        reward = tf.expand_dims(np.float32(reward), 0)
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

            # Add random noise to search in action space
            #search_noise = self.mu + self.std*np.random.random(self.num_actions)
            #search_noise = tf.convert_to_tensor(search_noise, dtype=tf.float32)

            #action = tf.math.add(action, search_noise)

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

def state_callback(state_msg):
    global s0, a0
    action = Float32MultiArray()
    s0 = state_msg.data
    a0 = agent_out(agents, actor_model, s0)
    send_action(pub, action, a0)

def reward_callback(reward_msg):
    global r
    r = reward_msg.data

def done_callback(done_msg):
    global done
    done = done_msg.data


if __name__ == '__main__':
    try:
        ''' Initialiaze agent'''
        agents = DDPG()
        agents.num_states = config.NUM_STATES
        agents.num_actions = config.NUM_ACTIONS
        agents.max_effort = config.MAX_EFFORT
        agents.scale_effort = config.SCALE_EFFORT

        actor_model = agents.Actor()
        critic_model = agents.Critic()
        #Restor last weights if PANDA approach is true
        if config.PANDA == True:
            actor_model.load_weights('./checkpoints/actor')
            critic_model.load_weights('./checkpoints/critic')
        
        critic_lr = config.CRITIC_LR
        actor_lr = config.ACTOR_LR

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        ''' Initialize global variables'''
        s0 = np.zeros((1, agents.num_states), dtype =np.float32)
        a0 = np.array(0, dtype = np.float32)
        r = np.array(0, dtype = np.float32)
        start = Bool()
        done = False
        TS = config.TIME_STEP #set same time step in simulink
        i, j = 0
        episode = 0
        states, actions, total_trajectory = []

        ''' Preparing for tensorboard'''

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'src/python/gradient_tape/' + current_time + '/training'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        ''' Push the agent in the network'''
        rospy.init_node('Agent')
        pub = rospy.Publisher('agent_action', Float32MultiArray, queue_size=10)
        start_simu = rospy.Publisher('start_simulation', Bool, queue_size=10)


        while rospy.is_shutdown() == False or episode < config.MAX_EPISODE:
            #Starting simulation with simulink
            if j > 0 and not(s0 == np.zeros((1, agents.num_states))):
                start.data = 0
                start_simu.publish(start)
            else:
                start.data = 1
                start_simu.publish(start)
                print('[INFO] Starting simulation in simulink')

            j = j + 1
            start.data = 0
            start_simu.publish(start)

            # Listen from simulink
            rospy.Subscriber("state", Float32MultiArray, state_callback)
            rospy.Subscriber("reward", Float32, reward_callback)
            rospy.Subscriber('is_done', Bool, done_callback)
            if done == False:
                if i < 2: 
                    #save states and actions at time t, t+1
                    states.append(s0)
                    actions.append(a0)
                    i = i +1
                else:
                    #fill trajectory
                    traj = np.array([states[0], actions[0], states[1], actions[1], r])
                    total_trajectory.append(traj)
                    states = []
                    actions = []
                    i = 0
                rospy.sleep(TS)
            else:
                done = False
                print('Goal is reached or episode is finish. Stopping simulation...')
                print('Starting learning for episode ', episode)
                trajectories = np.array(total_trajectory)
                loss_a, loss_c = []
                for i in range(trajectories.shape[0]):
                    aloss, closs = agents.update(trajectories[i][0],trajectories[i][1], trajectories[i][4], trajectories[i][2], trajectories[i][3] )
                    loss_a.append(aloss.numpy())
                    loss_c.append(closs.numpy())
                num_sample = trajectories.size[0]
                with train_summary_writer.as_default():
                    tf.summary.scalar('Actor loss', np.sum(np.array(loss_a))//num_sample, step=episode)
                    tf.summary.scalar('Critic loss', np.sum(np.array(loss_c))//num_sample, step=episode)
                    tf.summary.scalar('Reward',  np.sum(np.array(trajectories[:,4]))//num_sample, step=episode)    
                j = 0
                i = 0
                actor_model.save_weights('/checkpoints/actor')
                critic_model.save_weights('/checkpoints/critic')
                episode = episode + 1
            if episode == config.MAX_EPISODE:
                print('[INFO] Training on all episodes is finish. See results')
                print('Saving agents weights...')
                actor_model.save_weights('/checkpoints/actor')
                critic_model.save_weights('/checkpoints/critic')
                print('[INFO] Saving entire models...')
                actor_model.save('./model/actor')
                critic_model.save('./model/critic')
                break;
    except rospy.ROSInterruptException:
        pass