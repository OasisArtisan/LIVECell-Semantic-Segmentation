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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("annot_path", help="The path to the coco annotations json file", type=str)
    parser.add_argument("masks_path", help="The path to where to store the generated segmentation masks", type=str)
    
    args = parser.parse_args()
    
    coco=COCO(args.annot_path)
    
    assert len(coco.getCatIds()) == 1 # This scripts only generates a binary mask
    
    os.makedirs(args.masks_path, exist_ok=True)
    img_ids = coco.getImgIds()
    for img_id in tqdm.tqdm(img_ids, total=len(img_ids)):
        img = coco.loadImgs([img_id])[0]
        mask = np.zeros(shape=(img["height"], img["width"]))
        mask_pil = Image.fromarray(mask)
        polygons = list()
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(np.int32)
                # We have to do it one by one otherwise nested polygons may be subtracted.
                mask = cv2.fillPoly(mask, [poly], color=(255,255,255)) 

        Image.fromarray((mask).astype("uint8")).save(os.path.join(args.masks_path, img["original_filename"]))