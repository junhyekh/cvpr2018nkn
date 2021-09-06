import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from fk_torch import FK




class Gru_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kp=0.7):
        super(Gru_model, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.cells=[]
        self.dropout=[]
        self.cells.append(nn.GRUCell(input_size, hidden_size)) 
        for i in range(num_layers-1):
            self.cells.append(nn.GRUCell(hidden_size, hidden_size))  
        for i in range(num_layers):
            self.dropout.append(nn.Dropout(p=1-kp))
        self.cells=nn.ModuleList(self.cells)
        self.dropout=nn.ModuleList(self.dropout)     
    
    def forward(self, x, initial_state):
        hidden=[]
        for i in range(self.num_layers):
            x=self.dropout[i](x)
            x=self.cells[i](x, initial_state[i])
            hidden.append(x)
        return hidden
        
class Generator(nn.Module):
    def __init__(self, hidden_size, num_layers, n_joints, batch_size, kp):
        super(Generator, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.n_joints=n_joints
        self.batch_size=batch_size
        self.enc_gru = Gru_model(70, hidden_size, num_layers, kp).to("cuda")
        self.dec_gru = Gru_model(645, hidden_size, num_layers, kp).to("cuda")
        self.fc=nn.Linear(hidden_size , 4 * (n_joints + 1))
        self.fk=FK()

    def forward(self, seqA_, skelA_, skelB_, max_len, parents, dmean, dstd, training=False):
        b_local = []
        b_global = []
        b_quats = []
        a_local = []
        a_global = []
        a_quats = []

        statesA_AB = []
        statesB_AB = []
        statesA_BA = []
        statesB_BA = []
        
        for _ in range(self.num_layers):
            statesA_AB += [torch.zeros((self.batch_size, self.hidden_size), device="cuda")]
            statesB_AB += [torch.zeros((self.batch_size, self.hidden_size), device="cuda")]
            statesA_BA += [torch.zeros((self.batch_size, self.hidden_size), device="cuda")]
            statesB_BA += [torch.zeros((self.batch_size, self.hidden_size), device="cuda")]
        for t in range(max_len):
            """Retarget A to B"""       
            ptA_in = seqA_[:, t, :]
            statesA_AB= self.enc_gru(ptA_in, initial_state=statesA_AB)
            if t == 0:
                ptB_in = torch.zeros(self.batch_size, 3 * self.n_joints + 4, device="cuda")
            else:
                ptB_in = torch.cat([b_local[-1], b_global[-1]], dim=-1)

            ptcombined = torch.cat(
                [skelB_[:, 0, 3:], ptB_in, statesA_AB[-1]], dim=1)
            statesB_AB= self.dec_gru(ptcombined, statesB_AB)
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

                statesB_BA=self.enc_gru(ptB_in, initial_state=statesB_BA)

                if t == 0:
                    ptA_in = torch.zeros((self.batch_size, 3 * self.n_joints + 4),device="cuda")
                else:
                    ptA_in = torch.cat([a_local[-1], a_global[-1]], dim=-1)

                ptcombined = torch.cat(
                    [skelA_[:, 0, 3:], ptA_in, statesB_BA[-1]], dim=1)
                statesA_BA = self.dec_gru(ptcombined, initial_state=statesA_BA)
                angles_n_offset = self.fc(statesA_BA[-1])
                output_angles = torch.reshape(angles_n_offset[:, :-4],
                                        (self.batch_size, self.n_joints, 4))
                a_global.append(angles_n_offset[:, -4:])
                a_quats.append(self.normalized(output_angles))

                skel_in = torch.reshape(skelA_[:, 0, :], (self.batch_size, self.n_joints, 3))
                skel_in = skel_in * dstd + dmean

                output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
                output = torch.reshape(output, (self.batch_size, -1))
                a_local.append(output)

        return b_local, b_global, b_quats, a_local, a_global, a_quats

    def normalized(self, angles):
        lengths = torch.sqrt(torch.sum(torch.square(angles), axis=-1))
        return angles / lengths[..., None]


class Discriminator(nn.Module):
    def __init__(self, C, L, L_C):
        super(Discriminator, self).__init__()
        self.norm1=nn.BatchNorm1d(128)
        self.norm2=nn.BatchNorm1d(256)
        self.norm3=nn.BatchNorm1d(512)
        print(C, L, L_C)
        self.lrelu=nn.LeakyReLU()
        self.conv1=nn.Conv1d(in_channels=70, out_channels=64, kernel_size=4, stride=2, 
            padding=2)
        self.conv2=nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2,
            padding=1)
        self.conv3=nn.Conv1d(in_channels=128+63, out_channels=256, kernel_size=4, stride=2,
            padding=2)
        self.conv4=nn.Conv1d(in_channels=256+63, out_channels=512, kernel_size=4, stride=2,
            padding=1)
        self.conv5=nn.Conv1d(in_channels=512+63, out_channels=1, kernel_size=4, stride=2,
            padding=0)

    def forward(self, x, cond, kp, training=False):        
        x=x.permute(0,2,1)
        cond=cond.permute(0,2,1)
        x=F.dropout(x,1-kp, training=training)
        h0=self.lrelu(self.conv1(x))
        h1=self.lrelu(self.norm1(self.conv2(h0)))
        h1 = torch.cat([h1, torch.tile(cond, (1, 1, int(h1.shape[-1])))], dim=-2)
        h2=self.lrelu(self.norm2(self.conv3(h1)))
        h2 = torch.cat([h2, torch.tile(cond, (1, 1, int(h2.shape[-1])))], dim=-2)
        h3=self.lrelu(self.norm3(self.conv4(h2)))
        h3 = torch.cat([h3, torch.tile(cond, (1, 1, int(h3.shape[-1])))], dim=-2)
        logits = self.conv5(h3)
        return logits