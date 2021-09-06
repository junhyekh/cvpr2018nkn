"""Based on Daniel Holden code from http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing"""

import torch 


class FK(object):
  def __init__(self):
    pass

  def transforms_multiply(self, t0s, t1s):
    return torch.matmul(t0s, t1s)

  def transforms_blank(self, rotations):
    diagonal =  torch.diag(torch.ones(4, device="cuda"))[None, None, :, :]
    ts = torch.tile(diagonal,
                 (int(rotations.shape[0]),
                  int(rotations.shape[1]), 1, 1))

    return ts

  def transforms_rotations(self, rotations):
    q_length = torch.sqrt(torch.sum(torch.square(rotations), -1))
    qw = rotations[..., 0] / q_length
    qx = rotations[..., 1] / q_length
    qy = rotations[..., 2] / q_length
    qz = rotations[..., 3] / q_length
    """Unit quaternion based rotation matrix computation"""
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy],-1)
    dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx],-1)
    dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)],-1)
    m = torch.stack([dim0, dim1, dim2], -2)

    return m

  def transforms_local(self, positions, rotations):
    transforms = self.transforms_rotations(rotations)
    transforms = torch.cat([transforms, positions[:, :, :, None]], -1)
    zeros = torch.zeros(
        int(transforms.shape[0]),
         int(transforms.shape[1]), 1, 3, device="cuda")
    ones = torch.ones(int(transforms.shape[0]), int(transforms.shape[1]), 1, 1, device="cuda")
    zerosones = torch.cat([zeros, ones], dim=-1)
    transforms = torch.cat([transforms, zerosones], dim=-2)
    return transforms

  def transforms_global(self, parents, positions, rotations):
    locals = self.transforms_local(positions, rotations)
    globals = self.transforms_blank(rotations)

    globals = torch.cat([locals[:, 0:1], globals[:, 1:]], dim=1)
    globals = list(torch.split(globals, 1, dim=1))
    for i in range(1, positions.shape[1]):
      globals[i] = self.transforms_multiply(globals[parents[i]][:, 0],
                                            locals[:, i])[:, None, :, :]

    return torch.cat(globals, dim=1)

  def run(self, parents, positions, rotations):
    positions = self.transforms_global(parents, positions,
                                       rotations)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, None]
