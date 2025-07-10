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


class ValDatasetRegion(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, dataset_name, config, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num

        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))

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
        scene_locs[:scene_attr['locs'].shape[0]] = scene_attr['locs'][:self.max_obj_num]

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
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_anno(index)
        # obj_id = int(self.anno[index].get('obj_id', 0))
        obj_id = 0
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = int(self.anno[index].get('sqa_type', 0))
        if 'sqa_type' in self.anno[index]:
            type_info = self.anno[index]['sqa_type']
        elif 'eval_type' in self.anno[index]:
            type_info = self.anno[index]['eval_type']
        elif 'type_info' in self.anno[index]:
            type_info = self.anno[index]['type_info']
        if 'prompt' not in self.anno[index]:
            prompt = random.choice(view2cap_prompt)
            # prompt = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            prompt = self.anno[index]["prompt"]
        ref_captions = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else [
            self.anno[index]["utterance"]
        ]

        qid = self.anno[index]["item_id"] if "item_id" in self.anno[index] else 0
        if 'pos' in self.anno[index]:
            pos = torch.tensor(self.anno[index]['pos'])
        else:
            pos = torch.zeros((1, 7))
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, prompt, ref_captions, scene_id, qid, pred_id, type_info, pos


def val_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, prompts, ref_captions, scene_ids, qids, pred_ids, type_infos, pos = zip(
        *batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    obj_ids = torch.tensor(obj_ids)
    pred_ids = torch.tensor(pred_ids)
    pos = torch.stack(pos)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        "obj_ids": obj_ids,
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "type_infos": type_infos,
        "pos": pos
     # "ids": index
    }
