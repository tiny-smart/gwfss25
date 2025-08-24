#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
import sys
import numpy as np
import torch
from scipy import interpolate

"""
Usage:
  # download pretrained swin model:
  wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl
  # Then, use swin_tiny_patch4_window7_224.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if 'module' in obj.keys():
        obj = obj['module']
    if 'state_dict' in obj.keys():
        obj = obj['state_dict']
    
    new_obj = {}
    for k, v in obj.items():
        if 'relative_position_index' in k:
            continue
        if 'relative_position_bias_table' in k:
            rel_pos_bias = v
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, dst_patch_shape = 3972, (32, 32)
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 -
                                              1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens)**0.5)
            dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
            if src_size != dst_size:
                print('Position interpolate for %s from %dx%d to %dx%d' %
                      (k, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)
                
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q**(i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print('x = {}'.format(x))
                print('dx = {}'.format(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size,
                                                src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(
                            rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens),
                                             dim=0)
                v = new_rel_pos_bias
                print(v.shape)

        if sys.argv[3] == 'stage1':
          if 'backbone' in k:
            new_obj[k.replace('backbone', 'modelTeacher.backbone.model')] = v
          elif 'decode_head' in k:
            new_obj[k.replace('decode_head', 'modelTeacher.sem_seg_head')] = v
          else:
            new_obj['modelTeacher.backbone.model.'+k] = v
        elif sys.argv[3] == 'stage2':
          if 'backbone' in k:
            new_obj[k.replace('backbone', 'backbone.model')] = v
          elif 'decode_head' in k:
            new_obj[k.replace('decode_head', 'sem_seg_head')] = v
          else:
            new_obj['backbone.model.'+k] = v

    res = {"model": new_obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    # torch.save(res, sys.argv[2])
    
    print('\n'.join(new_obj.keys()))
