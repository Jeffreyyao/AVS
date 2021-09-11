import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import random
from spherical_harmonics import spherical_harmonics

# modeling the gravity acceleration of a point mass
# initialize the class with 5 parameters
# 1. the mass of the object the need to be modeled
# 2,3. max and min radius of the training data
# 4,5. batch size and epoches for training
class gravity_pinn():
    def __init__(self, mass, max_radius, min_radius):
        #tf.compat.v1.disable_eager_execution()
        self.train_size = 10000
        self.train_input = np.zeros((self.train_size,3), dtype=np.float)
        self.train_label = np.zeros((self.train_size,3), dtype=np.float)
        self.test_size = 10
        self.test_input = np.zeros((self.test_size,3), dtype=np.float)
        self.test_label = np.zeros((self.test_size,3), dtype=np.float)
        self.epoches = 100
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.diff_radius = self.max_radius-self.min_radius
        self.batch_size = 200
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
        self.input_min, self.input_max = None, None
        self.label_min, self.label_max = None, None

        # sperical harmonics parameters
        self.n_max = 2
        self.m_max = 2
        self.spherical_harmonics = spherical_harmonics(self.n_max,self.m_max,mass,min_radius)
        
    def get_true_acc_point_mass(self, r):
        return np.dot(-self.mu,r)/(np.linalg.norm(r)**3)
    
    def get_true_acc_SH(self,r):
        return self.spherical_harmonics.acceleration(r);

    def normalize_input(self, array):
        return (array-self.input_min)/(self.input_max-self.input_min)

    def denormalize_input(self, array):
        return array*(self.input_max-self.input_min)+self.input_min

    def normalize_label(self, array):
        return (array-self.label_min)/(self.label_max-self.label_min)

    def denormalize_label(self, array):
        return array*(self.label_max-self.label_min)+self.label_min
    
    # call this before calling train function
    def generate_data(self):
        # generate train data
        for i in range(self.train_size):
            # unit vector pointing at a random direction
            n = np.array([random()-0.5 for _ in range(3)])
            n /= np.linalg.norm(n)
            # multiply by magnitude to get position vector within desired range
            self.train_input[i] = np.array(n*(random()*self.diff_radius+self.min_radius))
            self.train_label[i] = self.get_true_acc_SH(self.train_input[i])

        # generate test data
        for i in range(self.test_size):
            n = np.array([random()-0.5 for _ in range(3)])
            n /= np.linalg.norm(n)
            self.test_input[i] = np.array(n*(random()*self.diff_radius+self.min_radius))
            self.test_label[i] = self.get_true_acc_SH(self.test_input[i])

        # min-max normalize train input and label
        self.input_min = np.min(self.train_input,axis=0)
        self.input_max = np.max(self.train_input,axis=0)
        self.label_min = np.min(self.train_label)
        self.label_max = np.max(self.train_label)
        self.train_input = self.normalize_input(self.train_input)
        self.train_label = self.normalize_label(self.train_label)
        self.test_input = self.normalize_input(self.test_input)
        self.test_label = self.normalize_label(self.test_label)

    def train(self):
        for _ in tqdm(range(self.epoches)):
            self.train_loss.reset_states()
            rand_indices = np.random.choice(self.train_size,self.batch_size)
            input_r = tf.convert_to_tensor(self.train_input[rand_indices])
            desired_acc = tf.convert_to_tensor(self.train_label[rand_indices])
            with tf.GradientTape() as tape_model:
                with tf.GradientTape() as tape_input_r:
                    tape_input_r.watch(input_r)
                    # neural net outputs potential from position vector
                    U = self.model(input_r)
                # taking gradient of U wrt input_r times -1
                # gravitational acceleration = -grad(gravitational potential)
                acc = tf.math.scalar_mul(-1.0,tape_input_r.gradient(U,input_r))
                loss = self.loss_object(desired_acc,acc)
            # taking the gradient of loss wrt model parameters
            gradients = tape_model.gradient(loss,self.model.trainable_variables)
            self.optimizer.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(gradients, self.model.trainable_variables) 
                if grad is not None)
            self.train_loss(loss)
            self.losses.append(self.train_loss.result().numpy())
        print(self.losses)
        _,ax = plt.subplots()
        ax.plot(np.arange(0,self.epoches,1),np.array(self.losses))
        plt.show()
    
    def test(self):
        input_r = tf.convert_to_tensor(self.test_input)
        with tf.GradientTape() as tape:
            tape.watch(input_r)
            U = self.model(input_r)
        acc = tf.math.scalar_mul(-1.0,tape.gradient(U,input_r))
        print("Comparing test data:")
        print("label/pinn output")
        for i in range(self.test_size):
            print(self.denormalize_label(self.test_label[i]),self.denormalize_label(acc[i].numpy()))


if __name__=="__main__":
    # earth radius: 6.371e6
    # altitude to space: 0.1e6
    model = gravity_pinn(mass=5.972e24,max_radius=6.471e6,min_radius=6.371e6)
    model.generate_data()
    model.train()
    model.test()