import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

#Tensorboard summaries
def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_'+ var_name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_'+ var_name, stddev)
        tf.summary.histogram('histogram_' + var_name, var)

class LSTMCompSeq2Seq(object):
    def __init__(self, encoder_steps, feature_size, decoder_steps, label_size, 
                 hidden_size, sess, auto_optimizer, pred_optimizer, reverse=False, unconditional_decode=False):
        
        self.reverse = reverse
        self.unconditional_decode = unconditional_decode
        self.sess = sess
        self.encoder_steps = encoder_steps
        self.feature_size = feature_size
        self.decoder_steps = decoder_steps
        self.label_size = label_size
        self.hidden_size = hidden_size
        
        self.auto_optimizer = auto_optimizer
        self.auto_lr = self.auto_optimizer._lr
        self.pred_optimizer = pred_optimizer
        self.pred_lr = self.pred_optimizer._lr
        
        #initializing placeholder
        inputs = tf.placeholder(tf.float32, [None, encoder_steps, feature_size], name='inputs_placeholder')
        labels = tf.placeholder(tf.float32, [None, decoder_steps, label_size], name='labels_placeholder')
        self.inputs = inputs
        self.labels = labels
        
        #reshape to [[batch_size, timestep],...,[b,t]]
        inputs = [tf.squeeze(step, [1]) for step in tf.split(inputs, encoder_steps, 1)]
        labels = [tf.squeeze(step, [1]) for step in tf.split(labels, decoder_steps, 1)]
        
        #init. LSTM Cells
        self._encoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(size) for size in hidden_size])
        self._decoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(size) for size in hidden_size])
        self._predictor_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(size) for size in hidden_size])
        
        #building graph
        encoder_outputs, encoder_state = self.encode(inputs)
        
        with tf.variable_scope('autoencoder'):
            decoder_outputs = self.decode(self._decoder_cell, inputs, self.feature_size, encoder_state, self.encoder_steps, self.unconditional_decode, 
                                          self.reverse, is_guided=False)
        #auto loss
        self.autoencoder_loss = tf.losses.mean_squared_error(self.inputs, decoder_outputs)
        self.auto_grad = self.auto_optimizer.compute_gradients(self.autoencoder_loss)
        self.auto_train_op = self.auto_optimizer.apply_gradients(self.auto_grad)
        with tf.name_scope('auto-train'):
            self.auto_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % v.name, g) for g, v in self.auto_grad])
        tf.summary.scalar('auto_loss', self.autoencoder_loss)
        
        ###############
        ## predictor ##
        ###############
        with tf.variable_scope('predictor'):
            predictor_outputs = self.decode(self._predictor_cell, labels, self.label_size, encoder_state, self.decoder_steps, self.unconditional_decode,
                                           reverse=False, is_guided=True)
        
        #pred loss
        self.predictor_loss = tf.losses.mean_squared_error(self.labels, predictor_outputs)        
        self.pred_grad = self.pred_optimizer.compute_gradients(self.predictor_loss)
        self.pred_grad = [(op, var) for op, var in self.pred_grad if op is not None]
        self.pred_train_op = self.pred_optimizer.apply_gradients(self.pred_grad)
        with tf.name_scope('pred-train'):
            self.pred_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % v.name, g) for g, v in self.pred_grad])
        tf.summary.scalar('pred_loss', self.predictor_loss)
        
    def train_autoencoder(self, inputs, learn_rate, merged):
        labels_holder = self.sess.run(tf.zeros([tf.shape(inputs)[0], self.decoder_steps, self.label_size], dtype=tf.float32))
        return self.sess.run([self.autoencoder_loss, merged, self.auto_grad_summ_op, self.auto_grad, self.auto_train_op], 
                             feed_dict={self.inputs:inputs, self.labels:labels_holder, self.auto_lr:learn_rate})
    
    def train_predictor(self, inputs, labels, learn_rate, merged):
        return self.sess.run([self.predictor_loss, merged, self.pred_grad_summ_op, self.pred_grad, self.pred_train_op], 
                            feed_dict={self.inputs:inputs, self.labels:labels, self.pred_lr:learn_rate})
    
    def encode(self, inputs):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_outputs, encoder_state = rnn.static_rnn(self._encoder_cell, inputs, dtype=tf.float32)
            #tensorboard dist. on encoder_state
            for i in range(len(encoder_state)):
                variable_summaries(encoder_state[i][0], 'enc_hidden_'+str(i))
                variable_summaries(encoder_state[i][1], 'enc_output_'+str(i))
            return encoder_outputs, encoder_state
            
    def decode(self, lstmcell, labels, label_size, encoder_state, future_steps, unconditional_decode, reverse, is_guided):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as var_scope:
            #initializing variables
            d_weights = tf.get_variable('d_weights', shape=[self.hidden_size[-1], label_size], 
                                        initializer=tf.truncated_normal_initializer, dtype=tf.float32)
            d_biases = tf.get_variable('d_biases', shape=[label_size], 
                                       initializer=tf.ones_initializer, dtype=tf.float32)
            
            #tensorboard summary
            variable_summaries(d_weights, 'dec_weights')
            variable_summaries(d_biases, 'dec_biases')

            if is_guided:
                #guided training, feeding labels at each timestep t
                decoder_state = encoder_state #using last generated input
                init_input = [tf.zeros([tf.shape(decoder_state[-1][1])[0], label_size], dtype=tf.float32)]
                    
                decoder_inputs_ = init_input + labels[:-1]
                decoder_outputs_, decoder_state = rnn.static_rnn(lstmcell, decoder_inputs_, 
                                                                initial_state=decoder_state, dtype=tf.float32)
                
                decoder_outputs = []
                for i in range(len(decoder_outputs_)):
                    if i > 0:
                        var_scope.reuse_variables()
                    decoder_outputs.append(tf.matmul(decoder_outputs_[i], d_weights) + d_biases)
                outputs = tf.transpose(tf.stack(decoder_outputs), [1, 0, 2])
                return outputs
            
            else:
                decoder_state = encoder_state #using encoder state
                decoder_outputs = []
                decoder_input_ = tf.zeros([tf.shape(decoder_state[-1][1])[0], label_size], 
                                          dtype=tf.float32) #initialize decoder input
                
                for step in range(future_steps):
                    if step > 0:
                        var_scope.reuse_variables()
                    decoder_input_, decoder_state = lstmcell(decoder_input_, decoder_state)
                    decoder_input_ = tf.matmul(decoder_input_, d_weights) + d_biases
                    decoder_outputs.append(decoder_input_)
                    
                if reverse and (future_steps==self.encoder_steps):
                    decoder_outputs = decoder_outputs[::-1] #reverse outputs' order
                outputs = tf.transpose(tf.stack(decoder_outputs), [1, 0, 2])
                return outputs
    
    def predict(self, inputs, predict_price=True):
        """ input as [batch_size, steps, num_features] """
        cell = self._predictor_cell if predict_price else self._decoder_cell
        steps = self.decoder_steps if predict_price else self.encoder_steps
        output_size = self.label_size if predict_price else self.feature_size
        scope = 'predictor' if predict_price else 'autoencoder'
        
        inputs = [tf.squeeze(step, [1]) for step in tf.split(inputs, self.encoder_steps, 1)]
        _, encoder_state = self.encode(inputs)
        
        with tf.variable_scope(scope):
            decoder_outputs = self.decode(cell, None, output_size, encoder_state, steps, self.unconditional_decode,
                                          reverse=self.reverse, is_guided=False)
        return self.sess.run(decoder_outputs)

def generate_train_sample(X, y, batch_size, X_time_step, y_time_step):
    if len(y.shape) != 2:
        y = np.reshape(y, (len(y),1))
    available_idxs = range(len(X) - X_time_step - y_time_step)
    start_batch_idxs = np.random.choice(available_idxs, batch_size, replace=False)
    X_batch_idxs = [list(range(i, i+X_time_step)) for i in start_batch_idxs]
    y_batch_idxs = [list(range(i+X_time_step, i+X_time_step+y_time_step)) for i in start_batch_idxs]
    return np.take(X, X_batch_idxs, axis=0), np.take(y, y_batch_idxs, axis=0)

def generate_test_sample(X, y, total_samples, start, X_time_step, y_time_step):
    X_batch_idxs = [list(range(start+i, start+i+X_time_step)) for i in range(total_samples)]
    y_batch_idxs = [list(range(start+i+X_time_step, start+i+X_time_step+y_time_step)) for i in range(total_samples)]
    return np.take(X, X_batch_idxs, axis=0), np.take(y, y_batch_idxs, axis=0)

def remove_overlap(data, steps):
    d = data[:steps-1]
    return np.append(d,(data[range(steps-1,len(data),steps)]))
