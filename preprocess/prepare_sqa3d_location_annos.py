import json
import numpy as np
import os
import nltk
import random
from tqdm import tqdm
import torch
import sys

sys.path.append('.')
from prompts.prompts import situation_grounding_prompt

anno_dir = 'annotations/sqa3d'
segmentor = 'mask3d'
version = ''


def convert_person_view(sentence):
    # first-person view to second-person view
    forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am': 'are'}

    def translate(word):
        if word.lower() in forms:
            return forms[word.lower()]
        return word

    result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
    return result.capitalize()


def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 0
    elif question[:2].lower() == 'is':
        return 1
    elif question[:3].lower() == 'how':
        return 2
    elif question[:3].lower() == 'can':
        return 3
    elif question[:5].lower() == 'which':
        return 4
    else:
        return 5     # others


def num_to_location_token(ori_num):
    ori_num = int(ori_num * 100) + 500
    if ori_num < 0:
        ori_num = 0
    if ori_num > 999:
        ori_num = 999
    return f"<LOC{ori_num:03}>"


scan2axis_align = json.load(open('annotations/scannet/scans_axis_alignment_matrices.json'))

for split in ['train', 'val']:

    dist_05_count = 0
    dist_10_count = 0

    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    scan_ids = []
    new_annos = []
    question_file = os.path.join(anno_dir, f'v1_balanced_questions_{split}_scannetv2.json')
    with open(question_file, 'r', encoding='utf-8') as f:
        question_data = json.load(f)['questions']
    question_map = {}
    for item in question_data:
        question_map[item['question_id']] = {
            's': [item['situation']] + item['alternative_situation'],     # list of str
            'q': item['question'],     # str
        }

    anno_file = os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{split}_scannetv2.json')
    with open(anno_file, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)['annotations']
    for item in tqdm(anno_data):
        scan_ids.append(item['scene_id'])
        scene_id = item['scene_id']
        obj_id = 0
        # situation = random.choice(question_map[item['question_id']]['s'])
        situation = ' '.join(question_map[item['question_id']]['s'])
        question = question_map[item['question_id']]['q']
        question_type = get_sqa_question_type(question)
        # prompt = situation + ' ' + question + " Answer the question using a single word or phrase."
        prompt = random.choice(situation_grounding_prompt).replace('<description>', situation)

        pos = np.array([item['position']['x'], item['position']['y'], item['position']['z']]).reshape(1, 3)

        gt_loc_tokens = [num_to_location_token(x) for x in pos[0, :2]]
        caption = "<LOCATION> " + " ".join(gt_loc_tokens) + " </LOCATION>"

        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        min_dist, min_id = 100, -1

        # axis_align_matrix = np.array(scan2axis_align[scene_id], dtype=np.float32).reshape(4, 4)
        # pts = np.ones((1, 4))
        # pts[:, :3] = pos
        # pos = np.dot(pts, axis_align_matrix.transpose())[:, :3]

        dist = np.linalg.norm(instance_locs[:, :2] - pos[:, :2], axis=1)
        min_id = np.argmin(dist)
        min_dist = dist[min_id]

        # find the id with distance < 0.5
        # ids_below_threshold = np.where(dist < 1.0)[0]
        # print(ids_below_threshold)
        # distances_below_threshold = dist[ids_below_threshold]
        # print(distances_below_threshold)

        dist_1 = np.linalg.norm(scannet_locs[:, :2] - pos[:, :2], axis=1)
        obj_id = np.argmin(dist_1)
        min_dist = dist_1[obj_id]

        if min_dist < 0.5:
            dist_05_count += 1
        if min_dist < 1.0:
            dist_10_count += 1

        if split == "train":
            # if max_iou >= args.train_iou_thres:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": int(min_id),
            # "caption": f"<OBJ{min_id:03}>.",
                "caption": caption,
                "prompt": prompt,
            # "x": pos[0, 0],
            # "y": pos[0, 1]
            })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": int(obj_id),
            # "ref_captions": [f"<OBJ{min_id:03}>."],
                "ref_captions": [caption],
                "prompt": prompt,
            # "x": pos[0, 0],
            # "y": pos[0, 1]
            })

        # break

    with open(f"annotations/sqa3d_g_{split}.json", "w") as f:
        json.dump(new_annos, f, indent=4)

    print(f"dist < 0.5: {dist_05_count / len(new_annos)}")
    print(f"dist < 1.0: {dist_10_count / len(new_annos)}")
