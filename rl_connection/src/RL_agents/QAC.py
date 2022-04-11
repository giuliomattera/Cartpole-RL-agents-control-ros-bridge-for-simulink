import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class QAC():

    def __init__(self, GAMMA=0.99, EFFORT=1, STD = 0.2, CLR = 1e-3, ALR = 1e-4, BATCH = 8):

        super(QAC, self).__init__()
        self.num_states = 1
        self.num_actions = 1
        self.gamma = GAMMA
        self.max_effort = EFFORT
        self.std = STD
        self.CRITIC_LR = CLR
        self.ACTOR_LR = ALR
        self.BATCH = BATCH

    def Actor(self):

        initializer = tf.keras.initializers.GlorotNormal()

        inputs = tf.keras.layers.Input(shape=(self.num_states))

        hidden = tf.keras.layers.Dense(256, 
        activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(inputs)
        
        hidden = tf.keras.layers.Dense(256, 
        activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(hidden)

        hidden = tf.keras.layers.Dense(128, 
        activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(hidden)

        mu = tf.keras.layers.Dense(
            self.num_actions, activation="tanh")(hidden)

        std = tf.keras.layers.Dense(
        self.num_actions, activation="sigmoid")(hidden)
        
        return tf.keras.Model(inputs, [mu, std])

    def Critic(self):

        initializer = tf.keras.initializers.GlorotNormal()
        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states) )
        
        hidden = tf.keras.layers.Dense(256, 
        activation=tf.keras.layers.ReLU(),
         kernel_initializer= initializer)(state_input)

        hidden = tf.keras.layers.Dense(256, 
        activation=tf.keras.layers.ReLU(),
         kernel_initializer= initializer)(hidden)
        
        hidden = tf.keras.layers.Dense(128,
         activation=tf.keras.layers.ReLU(), 
        kernel_initializer= initializer)(hidden)
        
        # Predicted V(s)
        outputs = tf.keras.layers.Dense(1)(hidden)

        return tf.keras.Model(state_input, outputs)

    def initializer(self):

        actor_model = self.Actor()
        critic_model = self.Critic()

        critic_lr = self.CRITIC_LR
        actor_lr = self.ACTOR_LR

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr )
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        models = [critic_model, actor_model]
        optimizers = [critic_optimizer, actor_optimizer]

        return models, optimizers

    def update(self, models, optimizers, state, action, reward, next_state, next_action):

        # Models = [critic_model, actor_model] Optimizers = [C_opt, A_opt]

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

        # CRITIC TRAINING : MINIMIZE THE MSE OF TD 
        with tf.GradientTape() as tape:
            
            critic_v= models[0](state, training=True)  # V(s) baseline

            critic_loss = tf.math.reduce_mean(tf.math.square(reward - critic_v))

        critic_grad = tape.gradient(
                critic_loss, models[0].trainable_variables)

        critic_grad = [(tf.clip_by_norm(grad, 1)) for grad in critic_grad]
            
        optimizers[0].apply_gradients(
                zip(critic_grad, models[0].trainable_variables))

        # ACTOR TRAINING : GRADIENT ASCENT OF ACTION-LOG-PROB*BASELINE

        with tf.GradientTape() as tape:

            [mu, std] = models[1](state, training = True)
            
            prob = tfp.distributions.Normal(mu*self.max_effort, std*self.max_effort)
            
            action_probs = prob.prob(action)
                                    
            critic_v = models[0](state, training=True)
            
            actor_loss = tf.math.reduce_mean(tf.math.negative(tf.math.log(action_probs) * (reward-critic_v)))

        actor_grad = tape.gradient(
                actor_loss, models[1].trainable_variables)

        actor_grad = [(tf.clip_by_norm(grad, 1)) for grad in actor_grad]
                        
        optimizers[1].apply_gradients(
                zip(actor_grad, models[1].trainable_variables))

        return critic_loss, actor_loss

    def train(self, models, optimizers, total_trajectory):
        global avg_rew

        num_sample = len(total_trajectory)
        total_rew, atl, ctl = [], [], []

        if num_sample > self.BATCH:
            for mb in range(num_sample//self.BATCH):
                mini_state0 = []
                mini_action0 = []
                mini_state1 = []
                mini_action1 = []
                mini_reward = []
                for sample in range(self.BATCH):
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

                print(' Critic {} and Actor {} losses for batch {} / {}'.format(np.mean(closs), np.mean(aloss), mb+1, num_sample//self.BATCH))
                print(' Seen so far: ', ((mb + 1) * self.BATCH), ' /', num_sample)
                total_rew.append(sum(mini_reward)/len(mini_reward))
                atl.append(aloss)
                ctl.append(closs)
                
            avg_reward = np.mean(total_rew)
            avg_atl = np.mean(atl)
            avg_ctl = np.mean(ctl)
            print("Avg discounted reward Gt {} , Avg actor loss is {}, Avg critic loss is {}".format(avg_reward, avg_atl, avg_ctl))
        else:
            print('[WARNING] Few samples acquired. Skipping training phase. ')
            avg_reward = 0

        return avg_reward

    def make_action(self, actor_model, state):

        s_nn = np.array(state, dtype=np.float32)
        s_nn = s_nn.resahpe(len(s_nn), self.num_states)

        [mu, std] = actor_model(s_nn)

        prob = tfp.distributions.Normal(mu*self.max_effort, std*self.max_effort)

        a0 = prob.sample()

        a0 = np.float32(a0)
    
        return a0