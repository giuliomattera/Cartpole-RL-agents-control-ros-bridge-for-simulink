import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

class DDPG():

    def __init__(self, gamma=0.99, effort=1, mu=0, std=1):
        super(DDPG, self).__init__()
        self.num_states = 1
        self.num_actions = 1
        self.gamma = gamma
        self.max_effort = effort
        self.mu = mu
        self.std = std

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
        state = tf.expand_dims(state, 0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        next_state = tf.expand_dims(next_state, 0)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

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

class CartPole():
    def __init__(self, mass_pole = 0.1, mass_cart = 1, length = 0.5, \
        angle_limit = 0.6, x_limit = 2, time_step = 0.01):

        #Mass [kg], angles [rad], disp [m]
        self.mp = mass_pole
        self.mc = mass_cart
        self.l = length
        self.ts = time_step

        self.g = 9.81
        self.mass = self.mp+self.mc
        self.I = self.mp*self.l

        self.angle_limit = angle_limit
        self.x_limit = x_limit

        self.state = None

    def dynamics(self, F):

        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            F + self.I * (theta_dot ** 2) * sintheta
        ) / self.mass

        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.l * (4.0 / 3.0 - self.mp * costheta ** 2 / self.mass)
        )
        xacc = temp - self.I * thetaacc * costheta / self.mass

        #Euler integrator
        x = x + self.ts* x_dot
        x_dot = x_dot + self.ts * xacc
        theta = theta + self.ts * theta_dot
        theta_dot = theta_dot + self.ts * thetaacc

        self.state = np.array((x, x_dot, theta, theta_dot), dtype = np.float32)

        done = bool(
            x < -self.x_limit
            or x > self.x_limit
            or theta < -self.angle_limit
            or theta > self.angle_limit
        )

        if done:
            reward = - 0.2*F*x_dot
        else:
            reward = - 0.2*F*x_dot - 2*np.abs(theta-self.angle_limit) 
            - np.abs(x-self.x_limit)
        
        return reward

#clear logs folder
# run in the command window tensorboard --logdir logs/gradient_tape

''' Preparing for tensorboard'''

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'src/python/logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

''' Initialize the agent'''
agents = DDPG()
agents.num_states = 4
agents.num_actions = 1
agents.max_effort = 100

actor_model = agents.Actor()
critic_model = agents.Critic()

# Learning rate for actor-critic models
critic_lr = 1e-5
actor_lr = 1e-5
agents.gamma = 0.99
agents.mu = 0.0
agents.std = 0.1

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

''' Initialize the environment'''

envs = CartPole()

''' Simulation settings '''

Ts = envs.ts
time = 10 #s
simu_step = int(time//Ts)
episodes = 200
simu_loss_a = []
simu_loss_c = []
simu_rew = []

for episode in range(episodes):
    envs.state = 0.1*np.random.rand(agents.num_states)
    print('[INFO] Inizilize state at ', envs.state)
    rew_old = 0
    all_traj = np.zeros((1, 5), dtype=np.float32)
    loss_c = []
    loss_a = []
    rew_tot = []
    for sim in range(simu_step):
        s0 = envs.state #store s0
        s0 = np.array(envs.state, np.float32)
        s_nn = tf.expand_dims(envs.state, 0) #prepare state for NN
        a = actor_model(s_nn) #get a0
        a = a + a*agents.std + agents.mu  # searching in action space
        a = a*agents.max_effort 
        a0 = np.clip(a.numpy(), -agents.max_effort, agents.max_effort)
  
        #get reward and store it
        rew = envs.dynamics(a.numpy())
        rew_new = rew + rew_old*(agents.gamma**sim)
        rew_old = rew_new #reset last reward
        #store s1
        s1 = envs.state

        s_nn = tf.expand_dims(s1, 0)
        a1 = actor_model(s_nn)

        if sim ==0:
            print('[INFO] Learning for epoch ', episode+1)
        #1 pass in backpropagation in the network
        closs, aloss = agents.update(s0, a0, rew_new, s1, a1)

        loss_c.append(closs.numpy())
        loss_a.append(aloss.numpy())
        rew_tot.append(rew_new)
    # Computing the avg of losses
    simu_loss_a.append(np.sum(np.array(loss_a))//simu_step)
    simu_loss_c.append(np.sum(np.array(loss_c))//simu_step)
    simu_rew.append(np.sum(np.array(rew_tot))//simu_step)

    ''' Send to tensorboard the results'''
    
    with train_summary_writer.as_default():
            tf.summary.scalar('Actor loss', np.sum(np.array(loss_a))//simu_step, step=episode)
            tf.summary.scalar('Critic loss', np.sum(np.array(loss_c))//simu_step, step=episode)
            tf.summary.scalar('Reward',  np.sum(np.array(rew_tot))//simu_step, step=episode)
    
    print('[INFO] An episod is finish. Metrics of ', episode + 1, ' :')
    print('[INFO] Avg reward', np.sum(np.array(rew_tot))//simu_step)
    print('[INFO] Avg actor loss ', np.sum(np.array(loss_a))//simu_step)
    print('[INFO] Avg critic loss ', np.sum(np.array(loss_c))//simu_step)
    print('--------------------------------------------------')

''' Visualization '''

print('[INFO] Episodes are finish. Evaluating and plotting global metrics...')
plt.figure()
f, axes = plt.subplots(3,1)
x = np.linspace(1, len(simu_loss_c), len(simu_loss_c))
axes[0].plot((simu_loss_c))

axes[0].set_ylabel("Critic losses")
axes[1].plot((simu_loss_a))
axes[1].set_ylabel("Actor losses")

axes[2].plot((simu_rew))
axes[2].set_ylabel("Rewards")
