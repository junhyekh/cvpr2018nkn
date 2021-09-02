from tensorflow.python.keras.backend import dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from fk_torch import FK




class Gru_model(nn.Module):
    def __init__(self, hidden_size, num_layers, kp):
        super(Gru_model, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=nn.GRU(70, hidden_size, num_layers, dropout=1-kp, batch_first=True)
    
    def forward(self, input, hidden):
        return self.gru(input, hidden)
    
    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
class Generator(nn.Module):
    def __init__(self, hidden_size, num_layers, n_joints, batch_size, kp):
        super(Generator, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.n_joints=n_joints
        self.batch_size=batch_size
        self.enc_gru = Gru_model(hidden_size, num_layers, kp)
        self.dec_gru = Gru_model(hidden_size, num_layers, kp)
        self.fc=nn.Linear(hidden_size , 4 * (n_joints + 1))
        self.fk=FK()

    def forward(self, seqA_, skelA_, skelB_, max_len, parents, dmean, dstd, training=False):
        b_local = []
        b_global = []
        b_quats = []
        a_local = []
        a_global = []
        a_quats = []

        statesA_AB = self.enc_gru.initHidden(self.batch_size)
        statesB_AB = self.dec_gru.initHidden(self.batch_size)
        statesA_BA = self.dec_gru.initHidden(self.batch_size)
        statesB_BA = self.enc_gru.initHidden(self.batch_size)
        for t in range(max_len):
            """Retarget A to B"""       
            ptA_in = seqA_[:, t:t+1, :]

            _, statesA_AB= self.enc_gru(ptA_in, initial_state=statesA_AB)
            if t == 0:
                ptB_in = torch.zeros(self.batch_size, 1, 3 * self.n_joints + 4)
            else:
                ptB_in = torch.cat([b_local[-1], b_global[-1]], dim=-1)

            ptcombined = torch.cat(
                [skelB_[:, 0, 3:], ptB_in, statesA_AB[-1]], dim=1)
            _, statesB_AB= self.enc_gru(ptcombined, statesB_AB)
            angles_n_offset = self.fc(statesB_AB[-1])
            output_angles = torch.reshape(angles_n_offset[:, :-4],
                                        (self.batch_size, self.n_joints, 4))
            b_global.append(angles_n_offset[:, -4:])
            b_quats.append(self.normalized(output_angles))

            skel_in = torch.reshape(skelB_[:, 0, :], (self.batch_size, self.n_joints, 3))
            skel_in = skel_in * dstd + dmean
            output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
            output = torch.reshape(output, (self.batch_size, -1))
            b_local.append(output)
            """Retarget B back to A"""
            if training:
                ptB_in = torch.cat([b_local[-1], b_global[-1]], dim=-1)

                n= self.enc_gru(tf.expand_dims(ptB_in, 1), training=True, initial_state=statesB_BA)
                statesB_BA=n[1:]

                if t == 0:
                ptA_in = tf.zeros([self.batch_size, 3 * self.n_joints + 4])
                else:
                ptA_in = tf.concat([a_local[-1], a_global[-1]], axis=-1)

                ptcombined = tf.concat(
                    values=[skelA_[:, 0, 3:], ptA_in, statesB_BA[-1]], axis=1)
                n = self.dec_gru(tf.expand_dims(ptcombined, 1), training=True, initial_state=statesA_BA)
                statesA_BA=n[1:]
                angles_n_offset = self.fc(statesA_BA[-1])
                output_angles = tf.reshape(angles_n_offset[:, :-4],
                                        [self.batch_size, self.n_joints, 4])
                a_global.append(angles_n_offset[:, -4:])
                a_quats.append(self.normalized(output_angles))

                skel_in = tf.reshape(skelA_[:, 0, :], [self.batch_size, self.n_joints, 3])
                skel_in = skel_in * dstd + dmean

                output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
                output = tf.reshape(output, [self.batch_size, -1])
                a_local.append(output)

        return b_local, b_global, b_quats, a_local, a_global, a_quats