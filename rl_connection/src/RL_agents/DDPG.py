import tensorflow as tf
import numpy as np

class DDPG():

    def __init__(self, GAMMA=0.99, EFFORT=1, CLR = 1e-3, ALR = 1e-3, BATCH = 8, TAU = 1e-3,
    STD = 0.2, DT = 1e-3, THETA = 0.15):
        super(DDPG, self).__init__()
        self.num_states = 1
        self.num_actions = 1
        self.gamma = GAMMA
        self.max_effort = EFFORT
        self.CRITIC_LR = CLR
        self.ACTOR_LR = ALR
        self.BATCH = BATCH
        self.TAU = TAU
        self.THETA = THETA
        self.DT = DT
        self.NMEAN = np.zeros(self.num_action)
        self.STD = np.float(STD)*np.ones(self.num_action)

    def Actor(self):

        initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        input = tf.keras.layers.Input(shape=(None,self.num_states))
        
        hidden = tf.keras.layers.Dense(400, 
        activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(input)
        
        hidden = tf.keras.layers.Dense(300, 
        activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(hidden)

        outputs = tf.keras.layers.Dense(
            self.num_actions, activation="tanh")(hidden)
        
        outputs = tf.keras.layers.experimental.preprocessing.Rescaling(self.max_effort)(outputs)    
        
        return tf.keras.Model(input, outputs)

    def Critic(self):

        initializer = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states) )
        
        state_out_critic = tf.keras.layers.Dense(400, 
        activation=tf.keras.layers.ReLU(),
         kernel_initializer= initializer)(state_input)
        
        state_out_critic = tf.keras.layers.Dense(300, 
        activation=tf.keras.layers.ReLU(),
         kernel_initializer= initializer)(state_out_critic)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))

        action_out_critic = tf.keras.layers.Dense(300,
        activation=tf.keras.layers.ReLU(),
         kernel_initializer= initializer)(action_input)

        # Concatening 2 networks
        concat = tf.keras.layers.Concatenate()(
            [state_out_critic, action_out_critic]
            )

        out = tf.keras.layers.Dense(128, 
            activation=tf.keras.layers.ReLU(),
            kernel_initializer= initializer)(concat)
        
        # Predicted Q(s,a)
        outputs = tf.keras.layers.Dense(1)(out)

        return tf.keras.Model([state_input, action_input], outputs)

    def initializer(self):

        actor_model = self.Actor()
        critic_model = self.Critic()

        target_actor = self.Actor()
        target_critic = self.Critic()

        critic_lr = self.CRITIC_LR
        actor_lr = self.ACTOR_LR

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr )
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


        models = [critic_model, actor_model, target_critic, target_actor]
        optimizers = [critic_optimizer, actor_optimizer]

        return models, optimizers 

    def update_target(self, target_weights, weights):
            for (a, b) in zip(target_weights, weights):
                a.assign(b * self.TAU + a * (1 - self.TAU))   
     
    def update(self, models, optimizers, state, action, reward, next_state, next_action):

        state = np.array(state)
        state = state.reshape(len(state),self.num_states)

        next_state = np.array(next_state)
        next_state = next_state.reshape(len(next_state),self.num_states)
        
        action = np.float32(action)
        action = action.reshape(len(action),self.num_actions)

        next_action = np.float32(next_action)
        next_action = next_action.reshape(len(next_action),self.num_actions)

        reward = np.float32(reward)
        reward = reward.reshape(len(reward), 1)

        with tf.GradientTape() as tape:
                        
            target_actions = models[3](next_state, training = True)

            critic_q = models[0]([state, action], training=True)  # Q(s,a)

            y = reward + self.gamma * \
                models[2]([next_state, target_actions],
                             training=True)  #reward - Q target
            
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_q))
            
        critic_grad = tape.gradient(
                critic_loss, models[0].trainable_variables)
            
        critic_grad = [(tf.clip_by_norm(grad, 1)) for grad in critic_grad]

        optimizers[0].apply_gradients(
                zip(critic_grad, models[0].trainable_variables))

        with tf.GradientTape() as gradient:

            actions = models[1](state, training=True)

            critic_q = models[0]([state, actions], training=True)

            actor_loss = tf.math.reduce_mean(-critic_q)

        actor_grad = gradient.gradient(
                actor_loss, models[1].trainable_variables)
            
        actor_grad = [(tf.clip_by_norm(grad, 1)) for grad in actor_grad]
            
        optimizers[1].apply_gradients(
                zip(actor_grad, models[1].trainable_variables))

        self.update_target(models[3].variables, models[1].variables)
        self.update_target(models[2].variables, models[0].variables)

        return critic_loss, actor_loss

    def UONise(self, bn):
        global n
        n = bn + self.THETA *(self.NMEAN - bn)*self.DT + self.STD + np.sqrt(self.DT) * np.random.normal(size=self.NMEAN.shape)

        return n

    def make_action(self, actor_model, state):
        global n
        s_nn = np.array(state, dtype=np.float32)
        s_nn = tf.expand_dims(s_nn, 0) #prepare state for NN 

        n = self.UONoise(n)

        return np.float32(actor_model(s_nn) + n)

    def train(self, models, optimizers, total_trajectory):

        num_sample = len(total_trajectory)
        total_rew, atl, ctl = [], [], []

        if num_sample > self.BATCH:
            for mb in range(num_sample//self.BATCH):
                mini_state0 = []
                mini_action0 = []
                mini_state1 = []
                mini_action1 = []
                mini_reward = []
                for sample in range(self.BATCH): #Check this impl.
                    mini_state0.append(total_trajectory[mb*self.BATCH:self.BATCH*(mb+1)+1][sample][0])
                    mini_action0.append(total_trajectory[mb*self.BATCH:self.BATCH*(mb+1)+1][sample][1])
                    mini_reward.append(total_trajectory[mb*self.BATCH:self.BATCH*(mb+1)+1][sample][2])
                    mini_state1.append(total_trajectory[mb*self.BATCH:self.BATCH*(mb+1)+1][sample][3])
                    mini_action1.append(total_trajectory[mb*self.BATCH:self.BATCH*(mb+1)+1][sample][4])
                    
                closs, aloss = self.update(models, optimizers,
                                            mini_state0,
                                            mini_action0, 
                                            mini_reward,
                                            mini_state1, 
                                            mini_action1)
                
                total_rew.append(sum(mini_reward)/len(mini_reward))
                atl.append(aloss)
                ctl.append(closs)
                
            avg_reward = np.mean(total_rew)
            avg_atl = np.mean(atl)
            avg_ctl = np.mean(ctl)
            print("Avg episod reward {} , Avg actor loss is {}, Avg critic loss is {}".format(avg_reward, avg_atl, avg_ctl))
        else:
            print('[WARNING] Few samples acquired. Skipping training phase. ')
            avg_reward = 0

        return avg_reward