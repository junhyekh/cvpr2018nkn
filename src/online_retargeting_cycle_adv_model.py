import os

import numpy as np
import tensorflow as tf

from forward_kinematics import FK
from ops import gaussian_noise as gnoise
from ops import conv1d
from ops import lrelu
from ops import linear
from ops import qlinear
from ops import relu
from tensorflow import atan2
from tensorflow import asin

layer_norm = tf.keras.layers.LayerNormalization


class EncoderDecoderGRU(object):
  def __init__(self,
               batch_size,
               alpha,
               beta,
               gamma,
               omega,
               euler_ord,
               n_joints,
               layers_units,
               max_len,
               dmean,
               dstd,
               omean,
               ostd,
               parents,
               keep_prob,
               logs_dir,
               learning_rate,
               optim_name,
               margin,
               d_arch,
               d_rand,
               norm_type,
               is_train=True):

    self.n_joints = n_joints
    self.batch_size = batch_size
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.omega = omega
    self.euler_ord = euler_ord
    self.kp = keep_prob
    self.max_len = max_len
    self.learning_rate = learning_rate
    self.fk = FK()
    self.d_arch = d_arch
    self.d_rand = d_rand
    self.margin = margin
    self.fake_score = 0.5
    self.norm_type = norm_type
    self.layers_units=layers_units
    self.dstd=dstd
    self.dmean=dmean
    self.omean=omean
    self.ostd=ostd
    self.parents=parents

    self.gen=self.generator(layers_units)
    self.disc=self.discriminator(tf.concat([tf.zeros(max_len, 3 * n_joints + 4)[:,:-4], (tf.zeros(max_len, 3 * n_joints + 4)[:,:-4])[:-1, :]],-1),tf.zeros((max_len, 3 * n_joints)))

    self.goptimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta1=0.5, name="goptimizer")

    self.doptimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta1=0.5, name="doptimizer")
    self.writer  = tf.summary.create_file_writer(logs_dir)
    
    self.saver = tf.train.Saver()

  def generator(self, layers_units):
    enc_gru = self.gru_model(layers_units)
    dec_gru = self.gru_model(layers_units)

    seqA_ = tf.keras.layers.Input(
        shape=(self.max_len, 3 * self.n_joints + 4),
        name="seqA")
    skelA_ = tf.keras.layers.Input(
        shape=(self.max_len, 3 * self.n_joints),
        name="skelA_")
    skelB_ = tf.keras.layers.Input(
        shape=(self.max_len, 3 * self.n_joints),
        name="skelB")

    b_local = []
    b_global = []
    b_quats = []
    a_local = []
    a_global = []
    a_quats = []

    statesA_AB = ()
    statesB_AB = ()
    statesA_BA = ()
    statesB_BA = ()
    fc=tf.keras.layers.Dense(4 * (self.n_joints + 1))
    for units in layers_units:
      statesA_AB += (tf.zeros([self.batch_size, units]), )
      statesB_AB += (tf.zeros([self.batch_size, units]), )
      statesA_BA += (tf.zeros([self.batch_size, units]), )
      statesB_BA += (tf.zeros([self.batch_size, units]), )

    for t in range(self.max_len):
      """Retarget A to B"""       
      ptA_in = seqA_[:, t, :]

      _, statesA_AB = enc_gru([ptA_in, statesA_AB])
      if t == 0:
        ptB_in = tf.zeros([self.batch_size, 3 * self.n_joints + 4])
      else:
        ptB_in = tf.concat([b_local[-1], b_global[-1]], axis=-1)

      ptcombined = tf.concat(
          values=[skelB_[:, 0, 3:], ptB_in, statesA_AB[-1]], axis=1)
      _, statesB_AB = dec_gru([ptcombined, statesB_AB])
      angles_n_offset = fc(statesB_AB[-1])
      output_angles = tf.reshape(angles_n_offset[:, :-4],
                                  [self.batch_size, self.n_joints, 4])
      b_global.append(angles_n_offset[:, -4:])
      b_quats.append(self.normalized(output_angles))

      skel_in = tf.reshape(skelB_[:, 0, :], [self.batch_size, self.n_joints, 3])
      skel_in = skel_in * self.dstd + self.dmean

      output = (self.fk.run(self.parents, skel_in, output_angles) - self.dmean) / self.dstd
      output = tf.reshape(output, [self.batch_size, -1])
      b_local.append(output)
      """Retarget B back to A"""
      ptB_in = tf.concat([b_local[-1], b_global[-1]], axis=-1)

      _, statesB_BA = enc_gru([ptB_in, statesB_BA])

      if t == 0:
        ptA_in = tf.zeros([self.batch_size, 3 * self.n_joints + 4])
      else:
        ptA_in = tf.concat([a_local[-1], a_global[-1]], axis=-1)

      ptcombined = tf.concat(
          values=[skelA_[:, 0, 3:], ptA_in, statesB_BA[-1]], axis=1)
      _, statesA_BA = dec_gru([ptcombined, statesA_BA])
      angles_n_offset = fc(statesA_BA[-1])
      output_angles = tf.reshape(angles_n_offset[:, :-4],
                                [self.batch_size, self.n_joints, 4])
      a_global.append(angles_n_offset[:, -4:])
      a_quats.append(self.normalized(output_angles))

      skel_in = tf.reshape(skelA_[:, 0, :], [self.batch_size, self.n_joints, 3])
      skel_in = skel_in * self.dstd + self.dmean

      output = (self.fk.run(self.parents, skel_in, output_angles) - self.dmean) / self.dstd
      output = tf.reshape(output, [self.batch_size, -1])
      a_local.append(output)

      return tf.keras.Model(inputs=[seqA_, skelA_, skelB_], outputs=[b_local, b_global, b_quats, a_local, a_global, a_quats]) 

  # def mlp_out(self, input_, reuse=False, name="mlp_out"):
  #   out = qlinear(input_, 4 * (self.n_joints + 1), name="dec_fc")
  #   return out

  def gru_model(self, layers_units, rnn_type="GRU"):
    gru_cells = [tf.keras.layers.GRUCell(units, dropout=(1-self.kp)) for units in layers_units]
    gru_layer=tf.keras.layers.RNN(gru_cells, return_state=True)
    return gru_layer

  def discriminator(self, input_, cond_):
    x=tf.keras.layers.Input(shape=input_.get_shape())
    cond=tf.keras.layers.Input(shape=cond_.get_shape())
    drop=tf.keras.layers.Input(shape=(1,))
    norm=tf.keras.layers.BatchNormalization()
    lrelu=tf.keras.layers.LeakyReLU()
    if drop>0:
      x=tf.keras.layers.Dropout(0.3)(x)
    if self.d_arch == 0:
      h0=lrelu(tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', name="conv1d_h0")(x))
      h1=lrelu(norm(tf.keras.layers.Conv1D(256, 4, strides=2, padding='same', name="conv1d_h1")(h0)))
      h1 = tf.concat([h1, tf.tile(cond, [1, int(h1.shape[1]), 1])], axis=-1)
      h2=lrelu(norm(tf.keras.layers.Conv1D(512, 4, strides=2, padding='same', name="conv1d_h2")(h1)))
      h2 = tf.concat([h2, tf.tile(cond, [1, int(h2.shape[1]), 1])], axis=-1)
      h3=lrelu(norm(tf.keras.layers.Conv1D(1024, 4, strides=2, padding='same', name="conv1d_h3")(h2)))
      h3 = tf.concat([h3, tf.tile(cond, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = tf.keras.layers.Conv1D(1, 4, strides=2, name="logits", padding="valid")(h3)
    elif self.d_arch == 1:
      h0=lrelu(tf.keras.layers.Conv1D(64, 4, strides=2, padding='same', name="conv1d_h0")(x))
      h1=lrelu(norm(tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', name="conv1d_h1")(h0)))
      h1 = tf.concat([h1, tf.tile(cond, [1, int(h1.shape[1]), 1])], axis=-1)
      h2=lrelu(norm(tf.keras.layers.Conv1D(256, 4, strides=2, padding='same', name="conv1d_h2")(h1)))
      h2 = tf.concat([h2, tf.tile(cond, [1, int(h2.shape[1]), 1])], axis=-1)
      h3=lrelu(norm(tf.keras.layers.Conv1D(512, 4, strides=2, padding='same', name="conv1d_h3")(h2)))
      h3 = tf.concat([h3, tf.tile(cond, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = tf.keras.layers.Conv1D(1, 4, strides=2, name="logits", padding="valid")(h3)
    elif self.d_arch == 2:
      h0=lrelu(tf.keras.layers.Conv1D(32, 4, strides=2, padding='same', name="conv1d_h0")(x))
      h1=lrelu(norm(tf.keras.layers.Conv1D(64, 4, strides=2, padding='same', name="conv1d_h1")(h0)))
      h1 = tf.concat([h1, tf.tile(cond, [1, int(h1.shape[1]), 1])], axis=-1)
      h2=lrelu(norm(tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', name="conv1d_h2")(h1)))
      h2 = tf.concat([h2, tf.tile(cond, [1, int(h2.shape[1]), 1])], axis=-1)
      h3=lrelu(norm(tf.keras.layers.Conv1D(256, 4, strides=2, padding='same', name="conv1d_h3")(h2)))
      h3 = tf.concat([h3, tf.tile(cond, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = tf.keras.layers.Conv1D(1, 4, strides=2, name="logits", padding="valid")(h3)
    elif self.d_arch == 3:
      h0=lrelu(tf.keras.layers.Conv1D(16, 4, strides=2, padding='same', name="conv1d_h0")(x))
      h1=lrelu(norm(tf.keras.layers.Conv1D(32, 4, strides=2, padding='same', name="conv1d_h1")(h0)))
      h1 = tf.concat([h1, tf.tile(cond, [1, int(h1.shape[1]), 1])], axis=-1)
      h2=lrelu(norm(tf.keras.layers.Conv1D(64, 4, strides=2, padding='same', name="conv1d_h2")(h1)))
      h2 = tf.concat([h2, tf.tile(cond, [1, int(h2.shape[1]), 1])], axis=-1)
      h3=lrelu(norm(tf.keras.layers.Conv1D(128, 4, strides=2, padding='same', name="conv1d_h3")(h2)))
      h3 = tf.concat([h3, tf.tile(cond, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = tf.keras.layers.Conv1D(1, 4, strides=2, name="logits", padding="valid")(h3)
    else:
      raise Exception("Unknown discriminator architecture!!!")
    out=tf.reshape(logits, [self.batch_size, 1])
    return tf.keras.Model(inputs=[x, cond, drop], outputs=out)

  def cyc_loss(self, seqA_, seqB_, mask_):
    output_localA = self.localA
    target_seqA = seqA_[:, :, :-4]
    cycle_local_loss = tf.reduce_sum(
        tf.square(
            tf.multiply(mask_[:, :, None],
                        tf.subtract(output_localA, target_seqA))))
    cycle_local_loss = tf.divide(cycle_local_loss,
                                      tf.reduce_sum(mask_))

    cycle_global_loss = tf.reduce_sum(
        tf.square(
            tf.multiply(mask_[:, :, None],
                        tf.subtract(seqA_[:, :, -4:], self.globalA))))
    cycle_global_loss = tf.divide(cycle_global_loss,
                                        tf.reduce_sum(mask_))

    dnorm_offA_ = self.globalA * self.ostd + self.omean
    cycle_smooth = tf.reduce_sum(
        tf.square(
            tf.multiply(mask_[:, 1:, None],
                        dnorm_offA_[:, 1:] - dnorm_offA_[:, :-1])))
    cycle_smooth = tf.divide(cycle_smooth,
                                  tf.reduce_sum(mask_))
    """ INTERMEDIATE OBJECTIVE FOR REGULARIZATION """
    output_localB = self.localB
    target_seqB = seqB_[:, :, :-4]
    interm_local_loss = tf.reduce_sum(
        tf.square(
            tf.multiply(self.aeReg_[:, :, None] * mask_[:, :, None],
                        tf.subtract(output_localB, target_seqB))))
    interm_local_loss = tf.divide(
        interm_local_loss,
        tf.maximum(tf.reduce_sum(self.aeReg_ * mask_), 1))

    interm_global_loss = tf.reduce_sum(
        tf.square(
            tf.multiply(self.aeReg_[:, :, None] * mask_[:, :, None],
                        tf.subtract(seqB_[:, :, -4:], self.globalB))))
    interm_global_loss = tf.divide(
        interm_global_loss,
        tf.maximum(tf.reduce_sum(self.aeReg_ * mask_), 1))

    dnorm_offB_ = self.globalB * self.ostd + self.omean
    interm_smooth = tf.reduce_sum(
        tf.square(
            tf.multiply(mask_[:, 1:, None],
                        dnorm_offB_[:, 1:] - dnorm_offB_[:, :-1])))
    interm_smooth = tf.divide(interm_smooth,
                                    tf.maximum(tf.reduce_sum(mask_), 1))
    return cycle_local_loss, cycle_global_loss, interm_local_loss, interm_global_loss, cycle_smooth, interm_smooth 
  
  def twist_loss(self):
    rads = self.alpha / 180.0
    twist_loss1 = tf.reduce_mean(
        tf.square(
            tf.maximum(
                0.0,
                tf.abs(self.euler(self.quatB, self.euler_ord)) - rads * np.pi)))
    twist_loss2 = tf.reduce_mean(
        tf.square(
            tf.maximum(
                0.0,
                tf.abs(self.euler(self.quatA, self.euler_ord)) - rads * np.pi)))
    """Twist loss"""
    return 0.5 * (twist_loss1 + twist_loss2)
  
  def disc_loss(self, real_logits, fake_logits):
    L_disc_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=real_logits, labels=tf.ones_like(real_logits)))

    L_disc_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    return L_disc_real, L_disc_fake

  def gen_loss(self, fake_logits):
    L_gen = tf.reduce_sum(
          tf.multiply((1 - self.aeReg_),
                      tf.nn.sigmoid_cross_entropy_with_logits(
                          logits=fake_logits,
                          labels=tf.ones_like(fake_logits))))
    L_gen = tf.divide(L_gen,
                            tf.maximum(tf.reduce_sum(1 - self.aeReg_), 1))

    return L_gen

  @tf.function
  def train(self, realSeq_, realSkel_, seqA_, skelA_, seqB_, skelB_, mask_,
            step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      mask_=tf.convert_to_tensor(mask_)
      b_local, b_global, b_quats, a_local, a_global, a_quats=self.gen(seqA_, skelA_, skelB_)
      self.localB = tf.stack(b_local, axis=1)
      self.globalB = tf.stack(b_global, axis=1)
      self.quatB = tf.stack(b_quats, axis=1)
      self.localA = tf.stack(a_local, axis=1)
      self.globalA = tf.stack(a_global, axis=1)
      self.quatA = tf.stack(a_quats, axis=1)
      dmean = self.dmean.reshape([1, 1, -1])
      dstd = self.dstd.reshape([1, 1, -1])

      if self.d_rand:
        dnorm_seq = realSeq_[:, :, :-4] * dstd + dmean
        dnorm_off = realSeq_[:, :, -4:] * self.ostd + self.omean
        skel_ = realSkel_[:, 0:1, 3:]
      else:
        dnorm_seq = seqA_[:, :, :-4] * dstd + dmean
        dnorm_off = seqA_[:, :, -4:] * self.ostd + self.omean
        skel_ = skelA_[:, 0:1, 3:]

      diff_seq = dnorm_seq[:, 1:, :] - dnorm_seq[:, :-1, :]
      real_data = tf.concat([diff_seq, dnorm_off[:, :-1, :]], axis=-1)
      real_logits = self.disc([real_data, skel_, 1])
      real_out = tf.sigmoid(real_logits)
      seqB = tf.concat([self.localB, self.globalB], axis=-1)
      dnorm_seqB = seqB[:, :, :-4] * dstd + dmean
      dnorm_offB = seqB[:, :, -4:] * self.ostd + self.omean
      diff_seqB = dnorm_seqB[:, 1:, :] - dnorm_seqB[:, :-1, :]
      fake_data = tf.concat([diff_seqB, dnorm_offB[:, :-1, :]], axis=-1)
      if self.fake_score > self.margin:
        print("update D")
        fake_logits = self.disc([fake_data, skelB_[:, 0:1, 3:],1])
        fake_out = tf.sigmoid(fake_logits)
        cur_score=fake_out.mean()
        #cur_score = self.D_.eval(feed_dict=feed_dict).mean()
        self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
        self.L_disc_real, self.L_disc_fake=self.disc_loss(real_logits, fake_logits)
        d_loss=self.L_disc_real +self.L_disc_fake
        discriminator_gradients=disc_tape.gradient(d_loss, self.disc.trainable_variables)
        self.doptimizer.apply_gradients(zip(discriminator_gradients, self.disc.trainable_variables))
        

      print("update G")
      fake_logits_g=self.disc([fake_data, skelB_[:, 0:1, 3:],0])
      fake_out_g=tf.sigmoid(fake_logits_g)
      cur_score =fake_out_g.mean()
      self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score

      cycle_local_loss, cycle_global_loss, interm_local_loss, interm_global_loss, cycle_smooth, interm_smooth = self.cyc_loss(seqA_, seqB_, mask_)
      twist_loss=self.twist_loss()
      smoothness = 0.5 * (interm_smooth + cycle_smooth)
      overall_loss = (
          cycle_local_loss + cycle_global_loss +
          interm_local_loss + interm_global_loss +
          self.gamma * twist_loss + self.omega * smoothness)
      L_gen=self.gen_loss(fake_logits_g)
      L = self.beta * L_gen + overall_loss
      generator_gradients=disc_tape.gradient(L, self.gen.trainable_variables)
      self.goptimizer.apply_gradients(zip(generator_gradients, self.gen.trainable_variables))
      with self.writer.as_default():
        tf.summary.scalar("losses/cycle_local_loss",
                                            cycle_local_loss, step)
        tf.summary.scalar("losses/cycle_global_loss",
                                            cycle_global_loss, step)
        tf.summary.scalar("losses/interm_local_loss",
                                            interm_local_loss, step)
        tf.summary.scalar("losses/interm_global_loss",
                                              interm_global_loss, step)
        tf.summary.scalar("losses/twist_loss", twist_loss, step)
        tf.summary.scalar("losses/smoothness", smoothness, step)

        tf.summary.scalar("losses/disc_real", self.L_disc_real, step)
        tf.summary.scalar("losses/disc_fake", self.L_disc_fake, step)
        tf.summary.scalar("losses/disc_gen", L_gen, step)

    return self.L_disc_fake, self.L_disc_real, L_gen, overall_loss

  def predict(self, seqA_, skelA_, skelB_):
    
    b_local, b_global, b_quats, _, _, _=self.gen(seqA_, skelA_, skelB_)
    output = np.concatenate((b_local, b_global), axis=-1)
    return output, b_quats

  @tf.function
  def normalized(self, angles):
    lengths = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(angles), axis=-1))
    return angles / lengths[..., None]
    
  @tf.function
  def euler(self, angles, order="yzx"):
    q = self.normalized(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    if order == "xyz":
      ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
      ey = asin(tf.clip_by_value(2 * (q0 * q2 - q3 * q1), -1, 1))
      ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
      return tf.stack(values=[ex, ez], axis=-1)[:, :, 1:]
    elif order == "yzx":
      ex = atan2(2 * (q1 * q0 - q2 * q3),
                 -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
      ey = atan2(2 * (q2 * q0 - q1 * q3),
                 q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
      ez = asin(tf.clip_by_value(2 * (q1 * q2 + q3 * q0), -1, 1))
      return ey[:, :, 1:]
    else:
      raise Exception("Unknown Euler order!")

  def save(self, checkpoint_dir, step):
    disc_name = "discriminator.model-"+str(step)+".h5"
    gen_name = " generator.model-"+str(step)+".h5"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.disc.save(os.path.join(checkpoint_dir, disc_name))
    self.gen.save(os.path.join(checkpoint_dir, gen_name))

  def load(self, checkpoint_dir, disc_name=None, gen_name=None):
    print("[*] Reading checkpoints...")
    try:
      self.gen=tf.keras.models.load_model(os.path.join(checkpoint_dir, gen_name))
      self.disc=tf.keras.models.load_model(os.path.join(checkpoint_dir, disc_name))
      return True
    except:
      return False
