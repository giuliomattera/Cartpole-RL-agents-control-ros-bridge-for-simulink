#!/usr/bin/env python3
import numpy as np
import rospy
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from RL_agents import QAC, DDPG
import config
import datetime, time
from std_msgs.msg import Float32, Float32MultiArray, Bool

def send_action(pub, action_msg, action):
    action_msg.data = action
    pub.publish(action_msg)

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

def communicate(agents, actor_model, publisher, total_trajectory, TYPE):

    global i, step, states, actions, traj, G_t, r, a0, s0, action

    rospy.Subscriber("state", Float32MultiArray, state_callback)
    rospy.Subscriber("reward", Float32, reward_callback)
    rospy.Subscriber("is_done", Bool, done_callback)

    a0 = agents.make_action(actor_model, s0)
    send_action(publisher, action, a0)

    if i < 2:
        #save states and actions at time t
        states.append(s0)
        actions.append(a0)
        G_t += r*(agents.gamma**step) #update discounted reward for metrics
        i = i +1

    else:

        #fill trajectory with time t,t+1
        if TYPE == 'DDPG':
            total_trajectory.append([states[0], actions[0], r, states[1], actions[1]])
        else:
            total_trajectory.append([states[0], actions[0], G_t, states[1], actions[1]])

        states = []
        actions = []
        i = 0
        step += 1
    
    rospy.Subscriber("is_done", Bool, done_callback)

if __name__ == '__main__':
    try:
        print('[INFO] Global variables initialization...')

        total_trajectory, tr = [], []

        TYPE = config.TYPE

        if TYPE == 'DDPG':
            agents = DDPG.DDPG()
            agents.num_states = 4
            agents.num_actions = 1
            agents.max_effort = 30
            agents.gamma = 0.98
            agents.std = 0.1
            agents.ACTOR_LR = 0.001
            agents.CRITIC_LR = 0.002
            agents.BATCH = 16
            models, optimizers = agents.initializer()

        else:
            agents = QAC.QAC()
            agents.num_states = 4
            agents.num_actions = 1
            agents.max_effort = 2
            agents.gamma = 0.98
            agents.std = 0.1
            agents.ACTOR_LR = 0.001
            agents.CRITIC_LR = 0.002
            agents.BATCH = 16
            models, optimizers = agents.initializer()

        s0 = [0]*agents.num_states
        a0 = np.array(0, dtype = np.float32)
        r = np.array(0, dtype = np.float32)
        G_t = np.array(0, dtype = np.float32)
        i, j, t, started, avg_rew, episode, step, done = 0 , 0, 0, 0, 0, 0, 0, False
        states, actions, traj, total_trajectory = [], [], [], []

        start = Bool()
        action = Float32()


        if config.PANDA == True:
            print('[INFO] Panda approach is used for learning. Importing last weights...')
            models[1].load_weights('./checkpoints/actor')
            models[0].load_weights('./checkpoints/critic')
            if TYPE == 'DDPG':
                models[2].load_weights('./checkpoints/target_actor')
                models[3].load_weights('./checkpoints/target_critic')

        else:
            print('[INFO] Trainig agents for the first time. Initializing agent s weights...')

        print('[INFO] Preparing for tensroboard...')


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './gradient_tape/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)


        print('[INFO] Pushing the agent in the network...')

        rospy.init_node('Agent')
        pub = rospy.Publisher('agent_action', Float32, queue_size=10)
        start_simu = rospy.Publisher('start_simulation', Bool, queue_size=10)

        if config.TRAIN == True:
            print('[INFO] Agent is set in learning mode.')
        else:
            print('[INFO] Agent is set in control mode.')

        start.data = 1
        start_simu.publish(start)

        #Starting simulation with simulink

        while rospy.is_shutdown() == False or episode < config.MAX_EPISODE:
            if j == 0: 
                print('[INFO] Asking for first simulation...')
                start.data = 1
                start_simu.publish(start)
                j = j + 1
            
            rospy.Subscriber("is_done", Bool, done_callback)
            rospy.Subscriber("reward", Float32, reward_callback)

            if done == False and r != 0:
                while done == False:
                    communicate(agents, models[1], pub, total_trajectory, TYPE)
                
            else:
                done = False
                loss_a, loss_c = [], []
                t = len(total_trajectory)

                if t != 0 and config.TRAIN == True:
                    print('[INFO] Simulation is finished. Starting learning from episode ', episode + 1)
                    avg_rew = agents.train(models, optimizers, total_trajectory)

                    total_trajectory = []
                    episode = episode + 1
                    print('[INFO] Sending results to tensorboard...')

                    with train_summary_writer.as_default():
                        tf.summary.scalar('Average Reward', avg_rew, step=episode)
                    
                    if episode < config.EPS_WARM and config.WARMUP == True:
                        optimizers[0].lr = exp_decay(episode, agents.CRITIC_LR,  -config.CLR_DECAY)
                        optimizers[1].lr = exp_decay(episode, agents.ACTOR_LR,  -config.ALR_DECAY)
                    else:
                        optimizers[0].lr = exp_decay(episode, agents.CRITIC_LR,  config.CLR_DECAY)
                        optimizers[1].lr = exp_decay(episode, agents.ACTOR_LR,  config.ALR_DECAY)
                    
                    if config.WARMUP == True:
                        if episode > config.EPS_WARM:
                            print('[INFO] Learning phase is finish. Saving agents weights...')
                            models[1].save_weights('./checkpoints/actor')
                            models[0].save_weights('./checkpoints/critic')
                        else:
                            print('[INFO] Learning finish. Weights are not saved because we are in Warmup phase.')
                    else:
                        print('[INFO[ Learning phase is finish. Saving agents weights...')
                        models[1].save_weights('./checkpoints/actor')
                        models[0].save_weights('./checkpoints/critic')
                        if episode == config.MAX_EPISODE:
                            print('[INFO] All episod are finish. Saving entire models...')
                            if config.TRAIN == True:
                                models[1].save('./model/actor')
                                models[0].save('./model/critic')
                            break;
                        print('------- Waiting for a new simulation -------')
                else: 
                    start.data = 1
                    start_simu.publish(start)
                    if j == 0:
                        print('------- Waiting for a new simulation -------')
                

                s0 = [0]*agents.num_states
                a0 = np.array(0, dtype = np.float32)
                r = np.array(0, dtype = np.float32)
                G_t = np.array(0, dtype = np.float32)
                avg_rew, step = 0, 0

                start.data = 1
                start_simu.publish(start)

    except rospy.ROSInterruptException:
        pass
