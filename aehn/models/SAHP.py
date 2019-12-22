from aehn.models import BaseModel, SelfAttention
import tensorflow as tf
from tensorflow.keras import layers


class SAHP(BaseModel):
    def train(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        lr = kwargs.get('lr')
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.learning_rate: lr
        }

        _, loss, pred_types, pred_time = sess.run([self.train_op, self.loss, self.pred_types, self.pred_time],
                                                  feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:]
        }
        return loss, preds, labels

    def predict(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs
        }
        loss, pred_types, pred_time = sess.run([self.loss, self.pred_types, self.pred_time],
                                               feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:]
        }
        return loss, preds, labels

    def __init__(self, model_config):
        # get hyperparameters from config
        super(SAHP, self).__init__(model_config)

        with tf.variable_scope('SAHP'):
            # 1. Embedding of input
            # shape -> [batch_size, max_len, hidden_dim]
            self.init_layers()

    def init_layers(self):
        self.type_seq_emb = layers.Embedding(self.process_dim + 1, self.hidden_dim, name='type_embedding')
        self.delta_layer = layers.Dense(self.process_dim, activation=tf.nn.softplus, name='delta_layer')
        self.mu_layer = layers.Dense(self.process_dim, activation=tf.nn.softplus, name='mu_layer')
        self.alpha_layer = layers.Dense(self.process_dim, activation=tf.nn.tanh, name='alpha_layer')
        return

    def intensity_layer(self, x_input, reuse=tf.AUTO_REUSE):
        """ Compute the imply lambdas.
        :param x_input: [batch_size, max_len, hidden_dim]
        :param reuse:
        :return: [batch_size, max_len, hidden_dim]
        """
        with tf.variable_scope('intensity_layer', reuse=reuse):
            attention_layer = SelfAttention(self.hidden_dim)

        with tf.name_scope('intensity_layer'):
            batch_size = tf.shape(x_input)[0]
            max_len = tf.shape(x_input)[1]

            # compute alpha
            # (batch_size, max_len, hidden_dim)
            h_states, _ = attention_layer.compute_attention_weight(x_input,
                                                                   x_input,
                                                                   pos_mask='self-right')
            # (batch_size, max_len, process_dim)
            lambdas = self.compute_lambda(h_states, self.dtimes_seq)

            if self.pred_method == 'loglikelihood':
                # sample lambdas for loss.
                dtimes_loss_samples = tf.linspace(start=0.0,
                                                  stop=1.0,
                                                  num=self.n_loss_integral_sample)

                # sample lambdas for prediction.

                # [batch_size, n_sample, max_len]
                dtimes_pred_samples = tf.linspace(start=0.0,
                                                  stop=self.max_time_pred,
                                                  num=self.n_pred_integral_sample)[None, :, None]
                # use loop to avoid memory explode
                # loop over max_len
                scan_shape = tf.stack([batch_size, self.n_pred_integral_sample, self.process_dim])
                init_shape_var = tf.zeros(scan_shape)

    def lambda_decay(self, mu, alpha, delta, time_decay):
        return tf.nn.softplus(mu + alpha * tf.exp(-delta * time_decay))

    def compute_lambda(self, h_states, dtimes):
        """
        :param h_states:  (batch_size, max_len, hidden_dim)
        :return: (batch_size, max_len, process_dim)
        """
        with tf.variable_scope('lambda_layer', reuse=tf.AUTO_REUSE):
            # (batch_size, max_len, 1)
            dtimes = tf.expand_dims(dtimes, axis=-1)

            # (batch_size, max_len, process_dim)
            mu = self.mu_layer(h_states)
            alpha = self.alpha_layer(h_states)
            delta = self.delta_layer(h_states)

            # (batch_size, max_len, process_dim)
            lambdas = self.lambda_decay(mu, alpha, delta, dtimes)
        return lambdas
