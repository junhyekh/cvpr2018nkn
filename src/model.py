import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from networks import *

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
                margin,
                d_arch,
                d_rand,
                norm_type):

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

        self.device='cuda:0'

        self.BCE=torch.nn.BCEWithLogitsLoss()

        self.gen=Generator(layers_units[0] ,len(layers_units), n_joints,
            batch_size, self.kp).to(self.device)
        self.gen.train()
        self.disc=Discriminator(C=59, L=3*n_joints+4, L_C=3*n_joints-3).to(self.device)
        self.disc.train()

        self.goptimizer = torch.optim.Adam(self.gen.parameters(),
                lr=self.learning_rate, betas=(0.5, 0.999))

        self.doptimizer = torch.optim.Adam(self.disc.parameters(),
                lr=self.learning_rate, betas=(0.5, 0.999))
        self.writer = SummaryWriter(logs_dir)
    
  

  # def mlp_out(self, input_, reuse=False, name="mlp_out"):
  #   out = qlinear(input_, 4 * (self.n_joints + 1), name="dec_fc")
  #   return out

    def cyc_loss(self, seqA_, seqB_, mask_, aeReg_):
        output_localA = self.localA
        target_seqA = seqA_[:, :, :-4]
        cycle_local_loss = torch.sum(
            torch.square(
                torch.multiply(mask_[:, :, None],
                            torch.subtract(output_localA, target_seqA))))
        cycle_local_loss = torch.divide(cycle_local_loss,
                                        torch.sum(mask_))

        cycle_global_loss = torch.sum(
            torch.square(
                torch.multiply(mask_[:, :, None],
                            torch.subtract(seqA_[:, :, -4:], self.globalA))))
        cycle_global_loss = torch.divide(cycle_global_loss,
                                            torch.sum(mask_))

        dnorm_offA_ = self.globalA * self.ostd + self.omean
        cycle_smooth = torch.sum(
            torch.square(
                torch.multiply(mask_[:, 1:, None],
                            dnorm_offA_[:, 1:] - dnorm_offA_[:, :-1])))
        cycle_smooth = torch.divide(cycle_smooth,
                                    torch.sum(mask_))
        """ INTERMEDIATE OBJECTIVE FOR REGULARIZATION """
        output_localB = self.localB
        target_seqB = seqB_[:, :, :-4]
        interm_local_loss = torch.sum(
            torch.square(
                torch.multiply(aeReg_[:, :, None] * mask_[:, :, None],
                            torch.subtract(output_localB, target_seqB))))
        interm_local_loss = torch.divide(
            interm_local_loss,
            torch.maximum(torch.sum(aeReg_ * mask_), 1))

        interm_global_loss = torch.sum(
            torch.square(
                torch.multiply(aeReg_[:, :, None] * mask_[:, :, None],
                            torch.subtract(seqB_[:, :, -4:], self.globalB))))
        interm_global_loss = torch.divide(
            interm_global_loss,
            torch.maximum(torch.sum(aeReg_ * mask_), 1))

        dnorm_offB_ = self.globalB * self.ostd + self.omean
        interm_smooth = torch.sum(
            torch.square(
                torch.multiply(mask_[:, 1:, None],
                            dnorm_offB_[:, 1:] - dnorm_offB_[:, :-1])))
        interm_smooth = torch.divide(interm_smooth,
                                        torch.maximum(torch.sum(mask_), 1))
        return cycle_local_loss, cycle_global_loss, interm_local_loss, interm_global_loss, cycle_smooth, interm_smooth 
  
    def twist_loss(self):
        rads = self.alpha / 180.0
        twist_loss1 = torch.mean(
            torch.square(
                torch.maximum(
                    0.0,
                    torch.abs(self.euler(self.quatB, self.euler_ord)) - rads * np.pi)))
        twist_loss2 = torch.mean(
            torch.square(
                torch.maximum(
                    0.0,
                    torch.abs(self.euler(self.quatA, self.euler_ord)) - rads * np.pi)))
        """Twist loss"""
        return 0.5 * (twist_loss1 + twist_loss2)
  
    def disc_loss(self, real_logits, fake_logits):
        L_disc_real = torch.mean(
           self.BCE(real_logits, Target=torch.ones_like(real_logits)))

        L_disc_fake = torch.mean(
           self.BCE(logits=fake_logits, labels=torch.zeros_like(fake_logits)))

        return L_disc_real, L_disc_fake

    def gen_loss(self, fake_logits, aeReg_):
        L_gen = torch.sum(
            torch.multiply((1 - aeReg_),
                        self.BCE(
                            logits=fake_logits,
                            labels=torch.ones_like(fake_logits))))
        L_gen = torch.divide(L_gen,
                            torch.maximum(torch.sum(1 - aeReg_), 1))

        return L_gen

    def set_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets=[nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad=requires_grad

    def train(self, realSeq_, realSkel_, seqA_, skelA_, seqB_, skelB_, aeReg_, mask_,
            step):
        self.gen.zero_grad()
        b_local, b_global, b_quats, a_local, a_global, a_quats=self.gen(seqA_, skelA_, skelB_, self.max_len, self.parents, self.dmean, self.dstd, training=True)
        self.localB = torch.stack(b_local, dim=1)
        self.globalB = torch.stack(b_global, dim=1)
        self.quatB = torch.stack(b_quats, dim=1)
        self.localA = torch.stack(a_local, dim=1)
        self.globalA = torch.stack(a_global, dim=1)
        self.quatA = torch.stack(a_quats, dim=1)
        dmean = self.dmean.reshape((1, 1, -1))
        dstd = self.dstd.reshape((1, 1, -1))

        if self.d_rand:
            dnorm_seq = realSeq_[:, :, :-4] * dstd + dmean
            dnorm_off = realSeq_[:, :, -4:] * self.ostd + self.omean
            skel_ = realSkel_[:, 0:1, 3:]
        else:
            dnorm_seq = seqA_[:, :, :-4] * dstd + dmean
            dnorm_off = seqA_[:, :, -4:] * self.ostd + self.omean
            skel_ = skelA_[:, 0:1, 3:]
        diff_seq = dnorm_seq[:, 1:, :] - dnorm_seq[:, :-1, :]
        real_data = torch.cat([diff_seq, dnorm_off[:, :-1, :]], dim=-1)
        seqB = torch.cat([self.localB, self.globalB], dim=-1)
        dnorm_seqB = seqB[:, :, :-4] * dstd + dmean
        dnorm_offB = seqB[:, :, -4:] * self.ostd + self.omean
        diff_seqB = dnorm_seqB[:, 1:, :] - dnorm_seqB[:, :-1, :]
        fake_data = torch.cat([diff_seqB, dnorm_offB[:, :-1, :]], axis=-1)
        if self.fake_score > self.margin:
            print("update D")
            self.set_grad(self.disc, True)
            self.disc.zero_grad()
            fake_logits = self.disc(fake_data.detach(), skelB_[:, 0:1, 3:], 0.7, training=True)
            fake_logits=torch.reshape(fake_logits, (self.batch_size, 1))
            real_logits = self.disc(real_data, skel_, 0, training=True)
            real_logits=torch.reshape(real_logits, (self.batch_size, 1))
            fake_out = torch.sigmoid(fake_logits)
            cur_score=fake_out.mean()
            #cur_score = self.D_.eval(feed_dict=feed_dict).mean()
            self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
            self.L_disc_real, self.L_disc_fake=self.disc_loss(real_logits, fake_logits)
            d_loss=self.L_disc_real +self.L_disc_fake
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 25)
            self.doptimizer.step()
            self.set_grad(self.disc, False)

        print("update G")
        fake_logits_g=self.disc(fake_data, skelB_[:, 0:1, 3:], self.batch_size)
        fake_out_g=torch.sigmoid(fake_logits_g)
        cur_score =fake_out_g.mean()
        self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score

        cycle_local_loss, cycle_global_loss, interm_local_loss, interm_global_loss, cycle_smooth, interm_smooth = self.cyc_loss(seqA_, seqB_, mask_, aeReg_)
        twist_loss=self.twist_loss()
        smoothness = 0.5 * (interm_smooth + cycle_smooth)
        overall_loss = (
            cycle_local_loss + cycle_global_loss +
            interm_local_loss + interm_global_loss +
            self.gamma * twist_loss + self.omega * smoothness)
        L_gen=self.gen_loss(fake_logits_g, aeReg_)
        L = self.beta * L_gen + overall_loss
        L.backward()
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 25)
        self.goptimizer.step()
        with self.writer.as_default():
            self.writer.add_scalar("losses/cycle_local_loss",
                                                cycle_local_loss, step)
            self.writer.add_scalar("losses/cycle_global_loss",
                                                cycle_global_loss, step)
            self.writer.add_scalar("losses/interm_local_loss",
                                                interm_local_loss, step)
            self.writer.add_scalar("losses/interm_global_loss",
                                                interm_global_loss, step)
            self.writer.add_scalar("losses/twist_loss", twist_loss, step)
            self.writer.add_scalar("losses/smoothness", smoothness, step)

            self.writer.add_scalar("losses/disc_real", self.L_disc_real, step)
            self.writer.add_scalar("losses/disc_fake", self.L_disc_fake, step)
            self.writer.add_scalar("losses/disc_gen", L_gen, step)

        return self.L_disc_fake, self.L_disc_real, L_gen, overall_loss

    def predict(self,seqA_, skelA_, skelB_):
        self.gen.eval()
        with torch.no_grad():
            b_local, b_global, b_quats, _, _, _=self.gen(seqA_, skelA_, skelB_, self.max_len, self.parents, self.dmean, self.dstd)
        output = torch.cat((b_local, b_global), dim=-1)
        return output.cpu().numpy(), b_quats.cpu().numpy()

    def normalized(self, angles):
        lengths = torch.sqrt(torch.sum(torch.square(angles), axis=-1))
        return angles / lengths[..., None]

    def euler(self, angles, order="yzx"):
        q = self.normalized(angles)
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]

        if order == "xyz":
            ex = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            ey = torch.asin(torch.clamp(2 * (q0 * q2 - q3 * q1), -1, 1))
            ez = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
            return torch.stack(values=[ex, ez], dim=-1)[:, :, 1:]
        elif order == "yzx":
            ex = torch.atan2(2 * (q1 * q0 - q2 * q3),
                        -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            ey = torch.atan2(2 * (q2 * q0 - q1 * q3),
                        q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            ez = torch.asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))
            return ey[:, :, 1:]
        else:
            raise Exception("Unknown Euler order!")

    def save(self, checkpoint_dir, step):
        torch.save(
            {
                'epoch': step,
                'discriminator': self.disc.state_dict(),
                'generator': self.gen.state_dict(),
                'dopt': self.doptimizer.state_dict(),
                'gopt': self.goptimizer.state_dict()
            }, checkpoint_dir
        )
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoints...")
        try:
            checkpoint=torch.load(checkpoint_dir)
            self.gen.load_state_dict(checkpoint['generator'])
            self.disc.load_state_dict(checkpoint['discriminator'])
            self.doptimizer.load_state_dict(checkpoint['dopt'])
            self.goptimizer.load_state_dict(checkpoint['gopt'])
            return True, checkpoint['epoch']
        except:
            return False