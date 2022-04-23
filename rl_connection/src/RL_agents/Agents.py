import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class DDPG():

    def __init__(self, GAMMA=0.99, UB=1, LB = -1, 
    CLR = 1e-3, ALR = 1e-3, BATCH = 8, TAU = 1e-3, STD = 0.2):

        self.num_states = 3
        self.num_actions = 1
        self.GAMMA = GAMMA
        self.UPPER_BOUND = UB
        self.LOWER_BOUND = LB
        self.CRITIC_LR = CLR
        self.ACTOR_LR = ALR
        self.BATCH = BATCH
        self.TAU = TAU
        self.buffer_capacity=int(100000)

        self.noise = tfp.distributions.Normal(0, STD)


    def getActor(self):
        inputs = tf.keras.layers.Input(shape=(self.num_states,))
        inputs_batch = tf.keras.layers.BatchNormalization()(inputs)
        out = tf.keras.layers.Dense(1024, activation="relu")(inputs_batch)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        outputs = tf.keras.layers.Dense(self.num_actions, activation="tanh",
        kernel_initializer= tf.random_uniform_initializer(minval=-0.1, maxval = 0.1, seed = 42))(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.UPPER_BOUND
        model = tf.keras.Model(inputs, outputs)
        return model


    def getCritic(self):
        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states))
        state_input_batch = tf.keras.layers.BatchNormalization()(state_input)
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_input_batch)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        action_out = tf.keras.layers.Dense(256, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])

        out = tf.keras.layers.Dense(256, activation="relu")(concat)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        outputs = tf.keras.layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    def make_action(self, actor_model, state):

        sampled_actions = actor_model(state)
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + self.noise.sample()

        legal_action = np.clip(sampled_actions, self.LOWER_BOUND, self.UPPER_BOUND)

        return legal_action

    def initialize(self):
        
        actor = self.getActor()
        target_actor = self.getActor()
        
        critic = self.getCritic()
        target_critic = self.getCritic()
        
        critic_optimizer = tf.keras.optimizers.Adam(self.CRITIC_LR)
        actor_optimizer = tf.keras.optimizers.Adam(self.ACTOR_LR)
        
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())
        
        self.getBuffer()
        
        return [actor, critic, target_actor, target_critic, critic_optimizer, actor_optimizer]

    
    def getBuffer(self):
        
        # Number of "experiences" to remember
        self.batch_size = self.BATCH
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, actor_model, critic_model, target_actor, target_critic,
               actor_optimizer, critic_optimizer,
               state_batch, action_batch, reward_batch, next_state_batch):
        
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + self.GAMMA * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.TAU + a * (1 - self.TAU))
        
    def learn(self, actor_model, critic_model, target_actor, target_critic,
               actor_optimizer, critic_optimizer):
        # Get sampling range
        if self.buffer_counter > self.BATCH:
            record_range = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

            self.update(actor_model, critic_model, target_actor, target_critic,
                    actor_optimizer, critic_optimizer, state_batch, action_batch, reward_batch, next_state_batch)
            
            self.update_target(target_actor.variables, actor_model.variables)
            self.update_target(target_critic.variables, critic_model.variables)


