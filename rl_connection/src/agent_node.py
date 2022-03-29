#!/usr/bin/env python3
import rospy
import tensorflow as tf
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, Bool
import datetime, time
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

        initializer = tf.keras.initializers.GlorotNormal()

        input = tf.keras.layers.Input(shape=(None,self.num_states))

        hidden = tf.keras.layers.Dense(128, 
        activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
        kernel_initializer= initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        )(input)

        hidden2 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
          kernel_initializer= initializer,
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(hidden)

        outputs = tf.keras.layers.Dense(
            self.num_actions, activation="tanh")(hidden)

        
        return tf.keras.Model(input, outputs)

    def Critic(self):

        initializer = tf.keras.initializers.GlorotNormal()

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states) )
        state_out_critic = tf.keras.layers.Dense(128, 
        activation=tf.keras.layers.LeakyReLU(alpha=0.1),
         kernel_initializer= initializer)(state_input)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))

        action_out_critic = tf.keras.layers.Dense(128,
        activation=tf.keras.layers.LeakyReLU(alpha=0.1),
         kernel_initializer= initializer)(action_input)

        # Concatening 2 networks
        concat = tf.keras.layers.Concatenate()(
            [state_out_critic, action_out_critic]
            )

        concat = tf.keras.layers.Dense(64,
         activation=tf.keras.layers.LeakyReLU(alpha=0.1), 
        kernel_initializer= initializer)(concat)

        out = tf.keras.layers.Dense(32, 
            activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            kernel_initializer= initializer,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(concat)

        # Predicted Q(s,a)
        outputs = tf.keras.layers.Dense(1)(out)

        return tf.keras.Model([state_input, action_input], outputs)

    def update(self, state, action, reward, next_state, next_action, batch):
        state = np.array(state)

        if batch == 1:
            state = state.reshape(len(state),self.num_states)
        else:
            state = state.reshape(1,4)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        next_state = np.array(next_state)
        if batch == 1:
            next_state = next_state.reshape(len(next_state),self.num_states)
        else:
            next_state = next_state.reshape(1,4)

        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        action = np.float32(action)
        if batch == 1:
            action = action.reshape(len(action),self.num_actions)
        else:
            action = tf.expand_dims(action, 0)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

        next_action = np.float32(next_action)
        if batch == 1:
            next_action = next_action.reshape(len(next_action),self.num_actions)
        else:
            next_action = tf.expand_dims(next_action, 0)
        next_action = tf.convert_to_tensor(next_action, dtype=tf.float32)

        reward = tf.expand_dims(np.float32(reward), 0)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as gradient:
            critic_q = critic_model([state, action], training=True)  # Q(s,a)

            y = reward + self.gamma * \
                critic_model([next_state, next_action],
                             training=True)  # Q(s',a')

            TD_error = y-critic_q

            critic_loss = tf.math.reduce_mean(
                tf.math.square(TD_error)
                )  # TD error

            critic_grad = gradient.gradient(
                critic_loss, critic_model.trainable_variables)

            critic_optimizer.apply_gradients(
                zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as gradient:
            action = actor_model(state, training=True)

            critic_q = critic_model([state, action], training=True)

            actor_loss = tf.math.reduce_mean(-critic_q)

            actor_grad = gradient.gradient(
                actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, actor_model.trainable_variables))

        return TD_error, actor_loss


def send_action(pub, action_msg, action):
    rate = rospy.Rate(1/TS)
    action_msg.data = action
    pub.publish(action_msg)
    rate.sleep()

def agent_out(agent, actor_model, state):
    global a0, eps

    s_nn = np.array(state, dtype=np.float32)
    s_nn = tf.expand_dims(s_nn, 0) #prepare state for NN
    a = actor_model(s_nn)
    a = a*agents.scale_effort #scale the action
    a = np.float32(a)
    rn = np.random.rand(1)
    
    if rn < eps and config.TRAIN == True:

        ar = np.random.normal(loc = config.MU, scale = agent.std)
        ar = np.float32(ar)

        ar = ar*agents.scale_effort #scale the action
        a0 = np.clip(ar, -agents.max_effort, agents.max_effort) #clip in max value
    else:
        
        a0 = np.clip(a, -agents.max_effort, agents.max_effort) #clip in max value
    
    return a0

def state_callback(state_msg):
    global s0
    s0 = state_msg.data

def reward_callback(reward_msg):
    global r
    r = reward_msg.data

def done_callback(done_msg):
    global done
    done = done_msg.data

def exp_decay(epoch, ini_value, decay):
   new_value = ini_value * np.exp(-decay*epoch)
   return new_value

if __name__ == '__main__':
    try:
        ''' Initialiaze agent'''
        agents = DDPG()
        agents.num_states = config.NUM_STATES
        agents.num_actions = config.NUM_ACTIONS
        agents.max_effort = config.MAX_EFFORT
        agents.scale_effort = config.SCALE_EFFORT
        agents.mu = config.MU
        agents.std = config.STD
        agents.gamma = 0.98

        actor_model = agents.Actor()
        critic_model = agents.Critic()

        #Restor last weights if PANDA approach is true

        if config.PANDA == True:
            print('[INFO] Importing last weights and last TD error...')
            actor_model.load_weights('./checkpoints/actor')
            critic_model.load_weights('./checkpoints/critic')
            best = np.load('./checkpoints/last_td.npy')
            print(' Last best : ', best)
        else:
            print('[INFO] Initializing agent s weights...')
            best = 1000000

        critic_lr = config.CRITIC_LR
        actor_lr = config.ACTOR_LR

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        ''' Initialize global variables'''
        eps = config.EPSILON
        s0 = [0]*config.NUM_STATES
        a0 = np.array(0, dtype = np.float32)
        r = np.array(0, dtype = np.float32)
        r_t = np.array(0, dtype = np.float32)
        start = Bool()
        action = Float32()
        done = False
        TS = config.TIME_STEP
        i, episode, t, started, j, EARLY = 0 , 0, 0, 0, 0, 0
        states, actions, traj, total_trajectory, = [], [], [], []

        BATCH = config.BATCH_SIZE
        ''' Preparing for tensorboard'''

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './gradient_tape/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        ''' Push the agent in the network'''

        rospy.init_node('Agent')
        pub = rospy.Publisher('agent_action', Float32, queue_size=10)
        start_simu = rospy.Publisher('start_simulation', Bool, queue_size=10)

        if config.TRAIN == True:
            print('Agent is in learning mode')
        else:
            print('Agent is in control mode')

        start.data = 1
        start_simu.publish(start)

        #Starting simulation with simulink

        while rospy.is_shutdown() == False or episode < config.MAX_EPISODE:

            if j == 0: 
                print('[INFO] Doing a new simulation in simulink...')
                start.data = 1
                start_simu.publish(start)
                j = j + 1
            
            rospy.Subscriber("is_done", Bool, done_callback)
            rospy.Subscriber("reward", Float32, reward_callback)

            if done == False and r != 0:

                rospy.Subscriber("state", Float32MultiArray, state_callback)
                rospy.Subscriber("reward", Float32, reward_callback)
                rospy.Subscriber("is_done", Bool, done_callback)
                a0 = agent_out(agents, actor_model, s0)
                send_action(pub, action, a0)

                if i < 2:
                    #save states and actions at time t
                    states.append(s0)
                    actions.append(a0)
                    i = i +1

                else:
                    r_t = r*agents.gamma**t + r_t #update discounted reward for metrics

                    #fill trajectory with time t,t+1
                    traj = [states[0], actions[0], states[1], actions[1], r]
                    total_trajectory.append(traj)
                    states = []
                    actions = []
                    i = 0
                    t = len(total_trajectory)
                rospy.sleep(TS)

            else:

                done = False
                loss_a, loss_c = [], []
                num_sample = len(total_trajectory)

                if config.TRAIN == True:

                    if not num_sample == 0:
                        print('[INFO] Episode is finish. Learning from episode ', episode, '/', config.MAX_EPISODE, '  with ', num_sample, ' samples..')
                        print('Final discounted reward: ', r_t)
                        #TO DO a batch training
                        if num_sample < BATCH:
                            print('Training without batch beacuse num of samples < batch size')
                            for i in range(num_sample):

                                aloss, closs = agents.update(total_trajectory[i][0],
                            total_trajectory[i][1], 
                            total_trajectory[i][4],
                            total_trajectory[i][2], 
                            total_trajectory[i][3],
                            0)

                                loss_a.append(aloss.numpy())
                                loss_c.append(closs.numpy())
                            avg_TD = np.sum(np.array(loss_c))/(num_sample)
                        else:
                            for mb in range(num_sample//BATCH):
                                mini_state0 = []
                                mini_action0 = []
                                mini_state1 = []
                                mini_action1 = []
                                mini_reward = []
                                for sample in range(BATCH):
                                    mini_state0.append(total_trajectory[mb*BATCH:BATCH*(mb+1)+1][sample][0])
                                    mini_action0.append(total_trajectory[mb*BATCH:BATCH*(mb+1)+1][sample][1])
                                    mini_state1.append(total_trajectory[mb*BATCH:BATCH*(mb+1)+1][sample][2])
                                    mini_action1.append(total_trajectory[mb*BATCH:BATCH*(mb+1)+1][sample][3])
                                    mini_reward.append(total_trajectory[mb*BATCH:BATCH*(mb+1)+1][sample][4])

                                aloss, closs = agents.update(mini_state0,
                                mini_action0, 
                                mini_reward,
                                mini_state1, 
                                mini_action0,
                                1)

                                loss_a.append(aloss.numpy())
                                loss_c.append(closs.numpy())

                                print(' TD error for batch ', mb, ' is ', closs.numpy())
                                print(' Seen so far: ', ((mb + 1) * BATCH), ' /', num_sample)
                            avg_TD = np.sum(np.array(loss_c))/(num_sample//BATCH)
                        print('The avg TD error : ', avg_TD)

                        with train_summary_writer.as_default():
                            tf.summary.scalar('Average of TD error', avg_TD, step=episode)
                            tf.summary.scalar('Discounted reward',  r_t/num_sample, step=episode)
                            tf.summary.scalar('Actor learning rate', actor_optimizer.lr, step = episode)
                            tf.summary.scalar('Critic learning rate', critic_optimizer.lr, step = episode)    
                        episode = episode + 1
                        j, i = 0, 0
                        total_trajectory = []
                        
                        if abs(avg_TD) < best and episode > 4:
                            print('TD error is deacresed. Learning phase is finish. Saving agents weights...')
                            actor_model.save_weights('./checkpoints/actor')
                            critic_model.save_weights('./checkpoints/critic')
                            best = abs(avg_TD)
                            EARLY = 0
                        elif episode == 0:
                            best = abs(avg_TD)
                        else:
                            print(' TD error is not decreased. The best is ', best, '. No saving chechpoint')
                            EARLY = EARLY + 1
                        print('-------------------------')
                    else:
                        start.data = 1
                        start_simu.publish(start)
                    
                start.data = 1
                start_simu.publish(start)
                s0 = [0]*config.NUM_STATES
                a0 = np.array(0, dtype = np.float32)
                r  = np.array(0, dtype = np.float32)
                r_t = np.array(0, dtype = np.float32)

                eps = exp_decay(episode, config.EPSILON, config.EPS_DECAY )
                if episode < config.EPS_WARM and config.WARMUP == True:
                    critic_optimizer.lr = exp_decay(episode, config.CRITIC_LR,  -config.CLR_DECAY)
                    actor_optimizer.lr = exp_decay(episode, config.ACTOR_LR,  -config.ALR_DECAY)
                else:
                    critic_optimizer.lr = exp_decay(episode, config.CRITIC_LR,  config.CLR_DECAY)
                    actor_optimizer.lr = exp_decay(episode, config.ACTOR_LR,  config.ALR_DECAY)


            if EARLY == config.EARLY_STOPPING:
                print('[INFO] Early stopping condition reached. Exit from simulation without saving checkpoints...')
                break;
            if episode == config.MAX_EPISODE:
                print('[INFO] All episod are finish. Saving entire models...')
                if config.TRAIN == True:
                    actor_model.save('./model/actor')
                    critic_model.save('./model/critic')
                    np.save('./checkpoints/last_td.npy', best)
                break;

    except rospy.ROSInterruptException:
        pass