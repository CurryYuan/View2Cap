import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from prompts.prompts import view2cap_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class TrainDatasetRegion(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, config, **kwargs):
        super().__init__()
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num

        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))

        if len(ann_list) > 4:
            sample_ratio = ann_list[-1]
            if sample_ratio < 1:
                self.anno = random.sample(self.anno, int(sample_ratio * len(self.anno)))

        self.feats = torch.load(feat_file, map_location='cpu')
        self.img_feats = torch.load(img_feat_file, map_location='cpu')

    def __len__(self):
        return len(self.anno)

    def get_anno(self, index):
        scene_id = self.anno[index]["scan_id"]
        obj_ids = self.anno[index]["obj_id"]
        # if self.feats is not None:
        #     scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        # else:
        #     scan_ids = set('_'.join(x.split('_')[:2]) for x in self.img_feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        unwanted_words = ["wall", "ceiling", "floor", "object", "item"]
        # for scan_id in scan_ids:
        #     if scan_id not in self.attributes:
        #         continue
        scene_attr = self.attributes[scene_id]
        # scene_locs = scene_attr["locs"]

        scene_locs = torch.zeros((self.max_obj_num, 6))
        scene_locs[:scene_attr['locs'].shape[0]] = scene_attr['locs']

        # obj_num = scene_attr['locs'].shape[0]
        obj_num = self.max_obj_num
        # obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
        obj_labels = scene_attr['objects'] if 'objects' in scene_attr else [''] * obj_num
        scene_feat = []
        scene_img_feat = []
        scene_mask = []

        for _id in range(self.max_obj_num):
            item_id = '_'.join([scene_id, f'{_id:02}'])
            if self.feats is None or item_id not in self.feats or _id not in obj_ids:
                # scene_feat.append(torch.randn((self.feat_dim)))
                scene_feat.append(torch.zeros(self.feat_dim))
                scene_mask.append(0)
            else:
                scene_feat.append(self.feats[item_id])
                scene_mask.append(1)
            if self.img_feats is None or item_id not in self.img_feats:
                # scene_img_feat.append(torch.randn((self.img_feat_dim)))
                scene_img_feat.append(torch.zeros(self.img_feat_dim))
            else:
                scene_img_feat.append(self.img_feats[item_id].float())
            # if scene_feat[-1] is None or any(x in obj_labels[_id] for x in unwanted_words):
            #     scene_mask.append(0)
            # else:
            # scene_mask.append(1)
        scene_feat = torch.stack(scene_feat, dim=0)
        scene_img_feat = torch.stack(scene_img_feat, dim=0)
        scene_mask = torch.tensor(scene_mask, dtype=torch.int)

        assigned_ids = torch.arange(self.max_obj_num)     # !!!
        return scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scan_id'] not in self.attributes:
            # print(f"{self.anno[index]['scene_id']} not in attribute file!")
            return self.__getitem__(random.randint(0, len(self.anno) - 1))
        # if "obj_id" in self.anno[index]:
        #     obj_id = int(self.anno[index]["obj_id"])
        # else:
        obj_id = random.randint(0, self.max_obj_num - 1)
        if 'prompt' not in self.anno[index]:
            question = random.choice(view2cap_prompt)     #.replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]

        caption = self.anno[index]["utterance"]

        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_anno(index)

        caption = update_caption(caption, assigned_ids)
        question = update_caption(question, assigned_ids)
        pos = torch.zeros((1, 9))
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, caption, question, pos


def train_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, captions, questions, pos = zip(
        *batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    obj_ids = torch.tensor(obj_ids)
    pos = torch.stack(pos)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
     # "detach_mask": batch_detach_mask,
        "obj_ids": obj_ids,
        "answers": captions,
        "questions": questions,
        "pos": pos
     # "ref_captions": ref_captions,
     # "ids": index
    }
