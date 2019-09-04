import tensorflow as tf
import numpy as np
import pickle
import os
import time
from matplotlib import pyplot as plt
import gym
import random
import mujoco_py


def set_seed(seed_number):
    os.environ['PYTHONHASHSEED']=str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.set_random_seed(seed_number)
    

def split_data(data):
    x, y = data['observations'], data['actions']
    n = x.shape[0]
    arr = np.arange(n)
    np.random.shuffle(arr)
    x, y = data['observations'][arr], data['actions'][arr]
    x_train, y_train = x[:int(0.6*n)], y[:int(0.6*n)]
    x_val, y_val = x[int(0.6*n):int(0.8*n)], y[int(0.6*n):int(0.8*n)]
    x_test, y_test = x[int(0.8*n):], y[:int(0.8*n)]
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_session(): # use with get_session() as sess: or sess = get_session()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

# define mlp nets
def model():
    state = tf.placeholder(tf.float32, [None, 376])
    action = tf.placeholder(tf.float32, [None, 1, 17])
    """
    with tf.variable_scope('conv'):
        conv1 = tf.contrib.layers.conv2d(state, num_outputs = 32, kernel_size = 8, stride = 4)
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 64, kernel_size = 4, stride = 2)
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 64, kernel_size = 3, stride = 1)
                
    conv3_flatten = tf.contrib.layers.flatten(conv3)
    """            
    with tf.variable_scope('fc'):
        fc1 = tf.contrib.layers.fully_connected(state, 1024)
        fc2 = tf.contrib.layers.fully_connected(state, 1024)
        fc3 = tf.contrib.layers.fully_connected(state, 784)
        out = tf.contrib.layers.fully_connected(fc3, 17, activation_fn=None)    
    
    out = tf.reshape(out, [tf.shape(state)[0], 1, 17])
    return state, action, out

def loss_function(pred, label):
    loss = tf.losses.mean_squared_error(pred, label)
    return loss

def optimizer(loss, lr = 1e-4):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_step


if __name__ == "__main__":
    with open('./expert_data/Humanoid-v2.pkl', 'rb') as handle:
        human = pickle.load(handle)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(human)    
    # training deep neural nets
    set_seed(1)
    sess = get_session()
    state, action, out = model()
    mean_loss = loss_function(out, action)
    train_step = optimizer(mean_loss)
        
    var= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    sess.run(tf.global_variables_initializer())    

    it_num = int(1e+5)
    batch_size = 100
    batch_size2 = 30
    epoch = 10

    for i in range(epoch):
        loss_his, loss_his2, loss_his3 = [], [], []
        for j in range(int(x_train.shape[0]/batch_size)):
            _, loss_train = sess.run([train_step, mean_loss], feed_dict={state:x_train[j*batch_size:(j+1)*batch_size]
                                                                                , action:y_train[j*batch_size:(j+1)*batch_size]})
            loss_his.append(loss_train)
        
            loss_val = sess.run([mean_loss], feed_dict={state:x_val[j*batch_size2:(j+1)*batch_size2]
                                                                                , action:y_val[j*batch_size2:(j+1)*batch_size2]})
            loss_test = sess.run([mean_loss], feed_dict={state:x_test[j*batch_size2:(j+1)*batch_size2]
                                                                                , action:y_test[j*batch_size2:(j+1)*batch_size2]})
            loss_his2.append(loss_val)
            loss_his3.append(loss_test)
        
        print(str(i)+' epoch: '+'train loss', loss_train)
        print('val loss: '+ str(np.mean(np.array(loss_his2)))+' test loss: ',str(np.mean(np.array(loss_his3))) )
    
    #plt.plot(loss_his)
    #plt.title('Loss train')
    #plt.show()
    
    env = gym.make('Humanoid-v2')
    num_rollouts = 200

    for i in range(num_rollouts):
        returns = []
        observations = []
        actions = []
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(out, feed_dict={state:obs.reshape(1, -1)})
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
        returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
    
    
    