"""
A small script to convert instance segmentation coco labels to semantic segmentation labels.
The script assumes that there is only one category.

Run the script with --help to learn about the usage.

Dependencies:
- pycocotools
- PIL
- tqdm
- opencv
- numpy

"""
from pycocotools.coco import COCO
import cv2
from PIL import Image
import tqdm
import numpy as np

import argparse
import os

def instance2semantic(annot_path, masks_path):
    print(f"Converting instance annotations in {annot_path} to semantic segmentation masks stored in {masks_path} directory")
    coco=COCO(annot_path)
    
    assert len(coco.getCatIds()) == 1 # This scripts only generates a binary mask
    
    os.makedirs(masks_path, exist_ok=True)
    img_ids = coco.getImgIds()
    for img_id in tqdm.tqdm(img_ids, total=len(img_ids)):
        img = coco.loadImgs([img_id])[0]
        mask = np.zeros(shape=(img["height"], img["width"]))
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(np.int32)
                # We have to do it one by one otherwise nested polygons may be subtracted.
                mask = cv2.fillPoly(mask, [poly], color=(255,255,255)) 

        Image.fromarray((mask).astype("uint8")).save(os.path.join(masks_path, img["original_filename"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--annot_path", help="The path to the coco annotations json file", type=str, default=None)
    parser.add_argument("--masks_path", help="The path to where to store the generated segmentation masks", type=str, default=None)
    
    args = parser.parse_args()

    if args.annot_path is None or args.masks_path is None:
        print("One or no arguments have been passed. Using default behavior which assumes dataset is in default location.")
        instance2semantic("./dataset/livecell_coco_train.json", "./dataset/images/livecell_train_val_masks")
        instance2semantic("./dataset/livecell_coco_val.json", "./dataset/images/livecell_train_val_masks")
        instance2semantic("./dataset/livecell_coco_test.json", "./dataset/images/livecell_test_masks")
    else:
        instance2semantic(args.annot_path, args.masks_path)
    
