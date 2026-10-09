import glob
import random
import torch
import os.path as op
import numpy as np
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv
from scipy.stats import pearsonr
import warnings
from scipy import stats
warnings.filterwarnings("ignore",category=stats.ConstantInputWarning)

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class MFQEv2Dataset(data.Dataset):
    """SCC dataset.

    For training data: LMDB is adopted. See create_lmdb for details.

    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            '/home/hzy/Documents/dataset/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            '/home/hzy/Documents/dataset/',
            self.opts_dict['lq_path']
        )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root,
            self.opts_dict['meta_info_fp']
        )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root,
            self.gt_root
        ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 1, 2, 3
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [2, 2, 2]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (7 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========

        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
        )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
        )
        TG = img_results[2]
        img_results1 = img_results[0:3]
        img_results2 = img_results[1:4]
        img_results3 = img_results[2:5]
        img0 = img_results[0].flatten()
        img1 = img_results[1].flatten()
        img2 = img_results[2].flatten()
        img3 = img_results[3].flatten()
        img4 = img_results[4].flatten()
        ps1,_ = pearsonr(img0,img2)
        ps2,_ = pearsonr(img1,img2)
        ps3,_ = pearsonr(img3,img2)
        ps4,_ = pearsonr(img4,img2)
        value = [ps1+ps2,ps2+ps3,ps3+ps4]
        max_value = max(value)
        max_index = value.index(max_value) #0,1,2
        if max_index == 0:
            img_finnal = img_results1
        elif max_index == 1:
            img_finnal = img_results2
        else:
            img_finnal = img_results3

        # to tensor
        img_TG = totensor(TG)
        img_finnal = totensor(img_finnal)
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_finnal[0:3], dim=0)
        img_lqs_long = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'lqlong': img_lqs_long,
            'gt': img_gt,  # ([RGB] H W)
            'TG': img_TG,
        }

    def __len__(self):
        return len(self.keys)


class VideoTestMFQEv2Dataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.

    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            '/media/hzy/data/Lowdelayyuv/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            '/media/hzy/data/Lowdelayyuv/',
            self.opts_dict['lq_path']
        )

        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [],
            'lq_indexes': [],
            'h': [],
            'w': [],
            'index_vid': [],
            'name_vid': [],
        }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            #          lq_name_vid = name_vid
            lq_name_vid = op.join('rec' + name_vid)  # add rec in here.
            w, h = map(int, name_vid.split('_')[-5].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-2])
            lq_vid_path = op.join(
                self.lq_root,
                lq_name_vid
            )
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
        )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
        ).astype(np.float32) / 255.  # (H W 1)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index],
                w=self.data_info['w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
            )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation
        TG = img_lqs[2]
        img_results1 = img_lqs[0:3]
        img_results2 = img_lqs[1:4]
        img_results3 = img_lqs[2:5]
        img0 = img_lqs[0].flatten()
        img1 = img_lqs[1].flatten()
        img2 = img_lqs[2].flatten()
        img3 = img_lqs[3].flatten()
        img4 = img_lqs[4].flatten()
        ps1,_ = pearsonr(img0,img2)
        ps2,_ = pearsonr(img1,img2)
        ps3,_ = pearsonr(img3,img2)
        ps4,_ = pearsonr(img4,img2)
        value = [ps1+ps2,ps2+ps3,ps3+ps4]
        max_value = max(value)
        max_index = value.index(max_value) #0,1,2
        if max_index == 0:
            img_finnal = img_results1
        elif max_index == 1:
            img_finnal = img_results2
        else:
            img_finnal = img_results3
        # to tensor
        img_finnal.append(img_gt)
        img_finnal = totensor(img_finnal)
        img_lqs_short = torch.stack(img_finnal[0:3], dim=0)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:5], dim=0)
        img_gt = img_finnal[-1]
        img_TG = totensor(TG)

        return {
            'lq': img_lqs_short,  # (T 1 H W)
            'lqlong': img_lqs,
            'gt': img_gt,  # (1 H W)
            'TG': img_TG,
            'name_vid': self.data_info['name_vid'][index],
            'index_vid': self.data_info['index_vid'][index],
        }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num

