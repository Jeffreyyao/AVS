import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import random

# modeling the gravity acceleration of a point mass
# initialize the class with 5 parameters
# 1. the mass of the object the need to be modeled
# 2,3. max and min radius of the training data
# 4,5. batch size and episodes for training
class gravity_pinn():
    def __init__(self, mass, max_radius, min_radius, batch_size, episodes):
        #tf.compat.v1.disable_eager_execution()
        self.episodes = episodes
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.batch_size = batch_size
        self.mass = mass
        self.G = 6.6743e-11
        self.mu = self.mass*self.G
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dense(128, activation='gelu'),
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dense(1)])
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.build(input_shape=(1,3))
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.losses = []
        
    def get_true_acc(self, r):
        return tf.math.scalar_mul(-self.mu,tf.convert_to_tensor(r))/(tf.norm(r)**3)
    
    def get_train_batch(self):
        desired_acc = [None for _ in range(self.batch_size)]
        input_r = [None for _ in range(self.batch_size)]
        diff_radius = self.max_radius-self.min_radius
        for i in range(self.batch_size):
            x = tf.constant(random()*diff_radius+self.min_radius,dtype=float)
            y = tf.constant(random()*diff_radius+self.min_radius,dtype=float)
            z = tf.constant(random()*diff_radius+self.min_radius,dtype=float)
            input_r[i] = [x,y,z]
            desired_acc[i] = self.get_true_acc(input_r[i])
        return tf.convert_to_tensor(input_r), tf.convert_to_tensor(desired_acc)

    def train(self):
        for _ in tqdm(range(self.episodes)):
            self.train_loss.reset_states()
            input_r,desired_acc = self.get_train_batch()
            with tf.GradientTape() as tape_model:
                with tf.GradientTape() as tape_input_r:
                    tape_input_r.watch(input_r)
                    # gravitational potential is modeled by the NN
                    U = self.model(input_r)
                # gravitational acceleration = -grad(gravitational potential)
                acc = tf.math.scalar_mul(-1.0,tape_input_r.gradient(U,input_r))
                loss = self.loss_object(desired_acc,acc)
            gradients = tape_model.gradient(loss,self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
            self.train_loss(loss)
            self.losses.append(self.train_loss.result().numpy())
        print(self.losses)
        _,ax = plt.subplots()
        ax.plot(np.arange(0,self.episodes,1),np.array(self.losses))
        plt.show()

if __name__=="__main__":
    model = gravity_pinn(mass=5.972e24,max_radius=6.371e10,min_radius=6.372e10,batch_size=100,episodes=100)
    model.train()