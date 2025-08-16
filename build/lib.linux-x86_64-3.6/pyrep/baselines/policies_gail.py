import numpy as np
import tensorflow as tf

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h


class GaussianPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            pi = fc(h2, 'pi', nact, act=lambda x: x, init_scale=0.01)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi + tf.random_normal(tf.shape(std), 0.0, 1.0) * std

        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.std = std
        self.logstd = logstd
        self.step = step
        self.value = value
        self.mean_std = tf.concat([pi, std], axis=1)
