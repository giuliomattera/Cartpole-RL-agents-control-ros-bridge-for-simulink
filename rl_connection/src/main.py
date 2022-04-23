#!/usr/bin/env python3
import numpy as np
import rospy
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from RL_agents import utils, Agents
import config
import datetime
import time
from std_msgs.msg import Float32, Float32MultiArray, Bool


def send_action(pub, action_msg, action):
    action_msg.data = action
    pub.publish(action_msg)


def state_callback(state_msg):
    global s0
    s0 = state_msg.data


def reward_callback(reward_msg):
    global r

    r = np.array(reward_msg.data, np.float32)


def done_callback(done_msg):
    global done

    done = done_msg.data


def exp_decay(epoch, ini_value, decay):

    new_value = ini_value * np.exp(-decay*epoch)

    return new_value


def communicate(agent):

    global i, step, states, actions, G_t, r, a0, s0, action

    rospy.Subscriber("state", Float32MultiArray, state_callback)
    rospy.Subscriber("reward", Float32, reward_callback)
    rospy.Subscriber("is_done", Bool, done_callback)
    
    s = tf.expand_dims(tf.convert_to_tensor(s0), 0)

    a0 = agents.make_action(actor_model, s)

    send_action(pub, action, a0)

    if i < 2:
        # save states and actions at time t
        states.append(s0)
        actions.append(a0)
        i = i + 1

    else:
        G_t += r # update total reward for metrics
        agent.record([states[0], actions[0], r, states[1]])
        states = []
        actions = []
        i = 0
    step += 1

    rospy.Subscriber("is_done", Bool, done_callback)
    rospy.sleep(0.1)


if __name__ == '__main__':
    try:

        global s0, a0, r, G_t, episode, done, step
        print('[INFO] Global variables initialization...')

        total_trajectory, tr = [], []
        #Using DDPG agent
        agents = Agents.DDPG(ALR = 1e-3, CLR = 1e-3, TAU = 1e-3, 
        GAMMA = 0.99, UB = 40, LB = -40, STD = 0.30)
        agents.num_states = 4
        agents.num_actions = 1
        agents.BATCH = 64
        agents.buffer_capacity = int(1000)


        actor_model, critic_model, target_actor, target_critic, critic_optimizer, actor_optimizer = agents.initialize()

        s0 = [0]*agents.num_states
        a0 = np.array(0, dtype=np.float32)
        r = np.array(0, dtype=np.float32)
        G_t = np.array(0, dtype=np.float32)
        i, j, episode, step, done = 0, 0, 0, 0, False
        states, actions = [], []

        start = Bool()
        action = Float32()

        if config.PANDA == True:
            print('[INFO] Panda approach is used for learning. Importing last weights...')
            actor_model.load_weights('./checkpoints/actor')
            critic_model.load_weights('./checkpoints/critic')
            target_actor.load_weights('./checkpoints/target_actor')
            target_critic.load_weights('./checkpoints/target_critic')

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

        # Starting simulation with simulink

        while rospy.is_shutdown() == False or episode < config.MAX_EPISODE:
            if j == 0:
                print('[INFO] Asking for first simulation...')
                start.data = 1
                start_simu.publish(start)
                j = j + 1

            rospy.Subscriber("is_done", Bool, done_callback)
            rospy.Subscriber("reward", Float32, reward_callback)
            rospy.sleep(0.1)

            if r!= 0 and done == False:
                while done == False:
                    communicate(agents)
                    rospy.Subscriber("is_done", Bool, done_callback)
                    #rospy.sleep(0.1)
                    if config.TRAIN == True:
                        agents.learn(actor_model, critic_model, target_actor, target_critic, 
                    actor_optimizer, critic_optimizer)

                print('[INFO] Episode {} is finished. In memory there are {} samples. Sending results to tensorboard'.format((episode + 1), 
                agents.buffer_counter))

                episode += 1
                
                if config.TRAIN == True:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Comulative avg discounted reward',
                                        G_t/step, step=episode)

                    if episode < config.EPS_WARM and config.WARMUP == True:
                        critic_optimizer.lr = exp_decay(episode, agents.CRITIC_LR,  -config.CLR_DECAY)
                        actor_optimizer.lr = exp_decay(episode, agents.ACTOR_LR,  -config.ALR_DECAY)
                    else:
                        critic_optimizer.lr = exp_decay(episode, agents.CRITIC_LR,  config.CLR_DECAY)
                        actor_optimizer.lr = exp_decay(episode, agents.ACTOR_LR,  config.ALR_DECAY)

                    if episode > config.MAX_EPISODE//2:
                        agents.STD = agents.STD/2
                        agents.BATCH = 2*agents.BATCH

                    if config.WARMUP == True:
                        if episode > config.EPS_WARM:
                            print('[INFO] Saving agents weights...')
                            actor_model.save_weights('./checkpoints/actor')
                            critic_model.save_weights('./checkpoints/critic')
                            target_actor.save_weights('./checkpoints/target_actor')
                            target_critic.save_weights('./checkpoints/target_critic')
                        else:
                            print('[INFO] Weights are not saved because we are in Warmup phase.')
                    else:
                        print('[INFO] Saving agents weights...')
                        actor_model.save_weights('./checkpoints/actor')
                        critic_model.save_weights('./checkpoints/critic')
                        target_actor.save_weights('./checkpoints/target_actor')
                        target_critic.save_weights('./checkpoints/target_critic')
                    
                    if episode == config.MAX_EPISODE:
                        print('[INFO] All episod are finish. Saving entire models...')
                        if config.TRAIN == True:
                            actor_model.save('./model/actor')
                            critic_model.save('./model/critic')
                            target_actor.save('./model/target_actor')
                            target_critic.save('./model/target_critic')
                            break;


                done = False

                s0 = [0]*agents.num_states
                a0 = np.array(0, dtype=np.float32)
                r = np.array(0, dtype=np.float32)
                G_t = np.array(0, dtype=np.float32)
                step = 0

                start.data = 1
                start_simu.publish(start)
                print('Asking for a new simulation.')
            else:
                start.data = 1
                start_simu.publish(start)
                done = False

    except rospy.ROSInterruptException:
        pass
