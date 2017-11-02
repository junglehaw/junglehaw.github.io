import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import collections
'''
import pandas as pd
import datetime
def download_price(id):
    df=get_price(id,start_date='2005-01-04',end_date=datetime.datetime.now())
    print(df)
    pd.DataFrame.to_csv(df, id + '.csv')


download_price('000001.XSHG')
download_price('399006.XSHE')
download_price('000905.XSHG')
download_price('000300.XSHG')
download_price('600809.XSHG')
download_price('600519.XSHG')
'''


class seq_batch(object):
    def __init__(self, data, time_steps, batch_size, target_index, lag, backward=False):
        self.data = data
        self.time_steps = time_steps
        self.batch_size = batch_size
        if backward:
            self.batch_start = len(self.data) % self.time_steps
        else:
            self.batch_start = 0

        self.target_index = target_index
        self.target_ifeats = np.logical_not(np.isin(np.arange(self.data.shape[1]),self.target_index))
        self.lag = lag


    def get_batch(self):
        # xs shape (50batch, 20steps)
        xs = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size).reshape((self.batch_size, self.time_steps))
        seq = self.data[xs]
        res = self.data[xs + self.lag][:, :, self.target_index]  # the high
        self.batch_start += self.time_steps
        # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
        # plt.show()
        # returned seq, res and xs: shape (batch, step, input)
        if self.batch_start > len(self.data) - self.batch_size*self.time_steps - self.lag:
            self.batch_start = np.random.randint(0, self.time_steps, 1)

        return [seq[:, :, self.target_ifeats], res, xs]

    def get_inference_batch(self):
        # xs shape (50batch, 20steps)
        xs = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size).reshape((self.batch_size, self.time_steps))
        seq = self.data[xs]
        res = self.data[xs + self.lag][:, :, self.target_index]  # the high
        self.batch_start += self.time_steps
        return [seq[:, :, 1:], res[:, :, np.newaxis], xs]

    def get_feats_batch(self):
        xs = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size).reshape((self.batch_size, self.time_steps))
        seq = self.data[xs]
        self.batch_start += self.time_steps
        return [seq[:, :, self.target_ifeats], xs]

class LSTMRegress(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def not_suspended(row):
    return not (row[0] == row[1] and row[0] == row[2] and row[0] == row[3])

def infer_show(n, save_path, session, saver, model, batcher, n_time_step=1, clf=True):
    saver.restore(session, save_path)
    for i in range(n):
        try:
            seq, res, xs = batcher.get_inference_batch()
        except IndexError:
            plt.pause(30)

        if i == 0:
            feed_dict = {model.xs: seq, model.ys: res}
        else:
            feed_dict = {model.xs: seq, model.ys: res, model.cell_final_state: state}

        state, pred = session.run([model.cell_final_state, model.pred], feed_dict=feed_dict)
        if clf:
            plt.clf()

        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[0:time_steps], 'b')
        plt.draw()
        plt.pause(2)

def predict_show(n, save_path, session, saver, model, batcher, clf=True):
    saver.restore(session, save_path)
    for i in range(n):
        try:
            seq, xs = batcher.get_feats_batch()
        except IndexError:
            p = np.max(xs) - batcher.lag
            for j in range(pred.shape[1]):
                a = plt.subplot(4, 1, j+1)
                a.axvline(x=p, color='r')
            plt.annotate(last_ts + ' +30d', xy=(p, 0))
            plt.tight_layout()
            plt.pause(30)

        if i == 0:
            feed_dict = {model.xs: seq, }
        else:
            feed_dict = {model.xs: seq, model.cell_final_state: state}

        state, pred = session.run([model.cell_final_state, model.pred], feed_dict=feed_dict)
        if clf:
            plt.clf()

        #plt.plot(xs[0, :], pred.flatten()[0:time_steps], 'b')
        for j in range(pred.shape[1]):
            a = plt.subplot(4, 1, j+1)
            a.plot(xs[0, :], pred[:, j], 'b--')
            a.set_title(mergedf.columns.values[batcher.target_index[j]])
        plt.draw()
    plt.pause(200)


def train(n, save_path, session, saver, model, batcher, clf=True):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', session.graph)
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(n):
        seq, res, xs = batcher.get_batch()
        if i == 0:
            feed_dict = {model.xs: seq, model.ys: res}
        else:
            feed_dict = {model.xs: seq, model.ys: res, model.cell_final_state: state}

        _, cost, state, pred = session.run([model.train_op, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)
        if clf:
            plt.clf()

        #plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[0:time_steps], 'b')
        for j in range(res[0].shape[1]):
            a = plt.subplot(4, 1, j+1)
            a.set_title(mergedf.columns.values[batcher.target_index[j]])
            a.plot(xs[0, :], res[0, :, j], 'r', xs[0, :], pred[:, j][0:time_steps], 'b')
        plt.draw()
        plt.pause(0.1 + i*0.005)

        if i % 20 == 0:
            print('cost', round(cost, 4))
            result = session.run(merged, feed_dict)
            writer.add_summary(result, i)

    saver.save(session, save_path)

batch_start = 0
time_steps = 300
batch_size = 1
#input_size = 15
#output_size = 2
cell_size = 32
learn_rate = 0.006
model_path = 'saver/600809'
inference = True
feats_common = ['open','close','high','low', 'total_turnover', 'volume']
feats_limits = ['limit_up', 'limit_down']
feats_target = ['ts'] + feats_common + feats_limits
col_names000 = ['ts'] + feats_common
usecols_qt = ['ts', 'open', 'total_turnover', 'volume']
usecols_target = ['ts', 'high', 'low', 'total_turnover', 'volume']

if __name__ == '__main__':
    as_list = pd.read_csv('as_list.csv', header=None)
    dx_list = pd.read_csv('dx_list.csv', header=None)
    dict_as = collections.OrderedDict()
    dict_dx = collections.OrderedDict()
    for sname in as_list.values[:, 0]:
        dict_as[sname] = pd.read_csv(sname + '.csv', names=feats_target, header=0, index_col='ts', usecols=usecols_target)
    for dxname in dx_list.values[:, 0]:
        dict_dx[dxname] = pd.read_csv(dxname + '.csv', names=col_names000, header=0, index_col='ts', usecols=usecols_qt)

    firstkey, mergedf = dict_as.popitem(last=False)
    while dict_as:
        k, v = dict_as.popitem(last=False)
        mergedf = mergedf.join(v, how='inner', rsuffix='-' + k)
    while dict_dx:
        k, v = dict_dx.popitem(last=False)
        mergedf = mergedf.join(v, how='inner', rsuffix='-' + k)

    for i in range(len(mergedf.columns.values)):
        print(i, mergedf.columns.values[i])

    last_ts = np.max(mergedf.index.values)
    #raw = merged.loc[:, feats_target[1]:].values
    #filtered = raw[np.array([not_suspended(row) for row in raw])][:, [0, 2, 4, 5, 8, 14]]
    filtered = mergedf.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    normv = scaler.fit_transform(filtered)
    plt.plot(normv)
    plt.show()
    #print(normv)
    target_ifeat = [0, 4, 8, 12]

    if inference:
        model = LSTMRegress(time_steps, normv.shape[1] - len(target_ifeat), len(target_ifeat), cell_size, 1)
        session = tf.Session()
        saver = tf.train.Saver()
        inference_batcher = seq_batch(normv, time_steps, batch_size, target_ifeat, 30, backward=True)
        #infer_show(200, model_path, session, saver, model, inference_batcher, clf=False)
        predict_show(200, model_path, session, saver, model, inference_batcher, clf=False)
    else:
        model = LSTMRegress(time_steps, normv.shape[1] - len(target_ifeat), len(target_ifeat), cell_size, batch_size)
        session = tf.Session()
        saver = tf.train.Saver()
        batcher = seq_batch(normv, time_steps, batch_size, target_ifeat, 30)
        train(200, model_path, session, saver, model, batcher, clf=False)
