import gym
import numpy as np
import tensorflow as tf
import collections
import random
import numpy as np
from sum_tree import SumTree
import math


class Memory(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size  # mini batch大小
        self.max_size = 2**math.floor(math.log2(max_size)) # 保证 sum tree 为完全二叉树
        self.beta = beta

        self._sum_tree = SumTree(max_size)

    def store_transition(self, s, a, r, s_, done):
        self._sum_tree.add((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        step = total // n_sample
        points_transitions_probs = []
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step - 1)
            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        points, transitions, probs = zip(*points_transitions_probs)

        # 计算重要性比率
        # max_impmortance_ratio = (n_sample * self._sum_tree.get_min())**-self.beta
        max_impmortance_ratio = (n_sample * (self._sum_tree.get_min() + 0.0001))
        max_impmortance_ratio = max_impmortance_ratio ** -self.beta

        importance_ratio = [(n_sample * probs[i])**-self.beta / max_impmortance_ratio
                            for i in range(len(probs))]

        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio

    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])


class DQN(object):
    def __init__(self, batch_size, gamma, lr, epsilon, replace_target_iter):
        self.env = gym.make('MountainCar-v0')
        self.env = self.env.unwrapped
        self.s_dim = self.env.observation_space.shape[0]  # 状态维度
        self.a_dim = self.env.action_space.n  # one hot行为维度

        self.memory = Memory(batch_size, 16384,  0.9)

        self.gamma = gamma
        self.epsilon = epsilon  # epsilon-greedy
        self.replace_target_iter = replace_target_iter  # 经历C步后更新target参数
        self._learn_step_counter = 0

    def choose_action(self, s, sess, q_eval_z, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_dim)
        else:
            q_eval_z = sess.run(q_eval_z, feed_dict={
                state: s[np.newaxis, :]
            })
            return q_eval_z.squeeze().argmax()

    def build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            x = tf.layers.dense(s, 20, tf.nn.relu, trainable=trainable)
            q_z = tf.layers.dense(x, self.a_dim, trainable=trainable)
        return q_z

    def build_model(self, lr):
        state = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        action = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        reward = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        next_state = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        done = tf.placeholder(tf.float32, shape=(None, 1), name='done')
        importance_ratio = tf.placeholder(tf.float32, shape=(None, 1), name='importance_ratio')

        q_eval_z = self.build_net(state, 'eval_net', True)
        q_target_z = self.build_net(next_state, 'target_net', False)

        # # argmax(Q)
        # max_a = tf.argmax(q_eval_z, axis=1)
        # one_hot_max_a = tf.one_hot(max_a, self.a_dim)
        #
        # y = r + gamma * max(q^)
        q_target = reward + self.gamma \
                   * tf.reduce_max(q_target_z, axis=1, keepdims=True) * (1 - done)
        q_target = tf.stop_gradient(q_target)

        q_eval = tf.reduce_sum(action * q_eval_z, axis=1, keepdims=True)
        # a_mask = tf.cast(self.a, tf.bool)
        # q_eval = tf.expand_dims(tf.boolean_mask(self.q_eval_z, a_mask), 1)

        td_error = tf.abs(q_target - q_eval)

        loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval)) * importance_ratio
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        param_target = tf.global_variables(scope='target_net')
        param_eval = tf.global_variables(scope='eval_net')

        # 将eval网络参数复制给target网络
        target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]
        return state, action, reward, next_state, done, optimizer, q_eval_z, q_target_z, target_replace_ops, \
                td_error, importance_ratio

    def train(self, train_steps=20001, batch_size=100, learning_rate=0.01, save_model_numbers=3):
        state, action, reward, next_state, done, \
            optimizer, \
            q_eval_z, q_target_z, \
            target_replace_ops, \
            td_error, importance_ratio = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            learn_step_counter = 0

            for epoch in range(500):
                s = self.env.reset()
                r_sum = 0
                while True:
                    a = self.choose_action(s, sess, q_eval_z, state, epsilon=self.epsilon)
                    s_, r, d, _ = self.env.step(a)

                    one_hot_action = np.zeros(self.a_dim)
                    one_hot_action[a] = 1

                    self.memory.store_transition(s, one_hot_action, [r], s_, [d])
                    # states, actions, rewards, next_states, dones = self.memory.get_mini_batches()

                    points, (states, actions, rewards, next_states, dones), importance_ratios \
                        = self.memory.get_mini_batches()

                    td_error_, _ = sess.run([td_error, optimizer],
                                            feed_dict={
                                            state: states,
                                            action: actions,
                                            reward: rewards,
                                            next_state: next_states,
                                            done: dones,
                                            importance_ratio: np.array([importance_ratios]).T
                                            })
                    self.memory.update(points, td_error_.squeeze(axis=1))

                    # 交换参数
                    learn_step_counter += 1
                    if learn_step_counter % self.replace_target_iter == 0:
                        sess.run(target_replace_ops)

                    r_sum += 1
                    if d:
                        print(epoch, r_sum)
                        if epoch > 490:
                            saver.save(sess, 'ckpt/mnist.ckpt', global_step=epoch)
                        break
                    s = s_

    def continue_train(self, epochs=50, learning_rate=0.01, save_model_numbers=3):
        state, action, reward, next_state, done, \
            optimizer, \
            q_eval_z, q_target_z, \
            target_replace_ops, \
            td_error, importance_ratio = self.build_model(learning_rate)
        saver = tf.train.Saver(max_to_keep=save_model_numbers)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            learn_step_counter = 0

            for epoch in range(epochs):
                s = self.env.reset()
                r_sum = 0
                while True:
                    a = self.choose_action(s, sess, q_eval_z, state, epsilon=self.epsilon)
                    s_, r, d, _ = self.env.step(a)

                    one_hot_action = np.zeros(self.a_dim)
                    one_hot_action[a] = 1

                    self.memory.store_transition(s, one_hot_action, [r], s_, [d])
                    # states, actions, rewards, next_states, dones = self.memory.get_mini_batches()

                    points, (states, actions, rewards, next_states, dones), importance_ratios \
                        = self.memory.get_mini_batches()

                    td_error_, _ = sess.run([td_error, optimizer],
                                            feed_dict={
                                                state: states,
                                                action: actions,
                                                reward: rewards,
                                                next_state: next_states,
                                                done: dones,
                                                importance_ratio: np.array([importance_ratios]).T
                                            })
                    self.memory.update(points, td_error_.squeeze(axis=1))

                    # 交换参数
                    learn_step_counter += 1
                    if learn_step_counter % self.replace_target_iter == 0:
                        sess.run(target_replace_ops)

                    r_sum += 1
                    if d:
                        print(epoch, r_sum)
                        if epoch > (epochs-10):
                            saver.save(sess, 'ckpt/mnist.ckpt', global_step=epoch)
                        break
                    s = s_

    def play(self):
        state, action, reward, next_state, done, \
            optimizer, \
            q_eval_z, q_target_z, \
            target_replace_ops, \
            td_error, importance_ratio = self.build_model(0.01)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_file = tf.train.latest_checkpoint('ckpt/')
            saver.restore(sess, model_file)

            steps = []
            for epoch in range(100):
                s = self.env.reset()
                step = 0
                while True:
                    # self.env.render()
                    a = self.choose_action(s, sess, q_eval_z, state, epsilon=0.1)
                    s_, r, d, _ = self.env.step(a)

                    step += 1
                    # print(step)
                    if d:
                        print(step)
                        steps.append(step)
                        break
                    s = s_
            print(np.mean(steps))


if __name__ == '__main__':
    agent = DQN(batch_size=128, gamma=0.99, lr=0.01, epsilon=0.1, replace_target_iter=300)
    # agent.train()
    # agent.continue_train(epochs=500, learning_rate=0.01)
    agent.play()

# 500 274.55


