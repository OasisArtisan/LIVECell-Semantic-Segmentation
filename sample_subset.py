"""
A script to sample a random small set from the livecell dataset.

The script assumes dataset files are in default locations as described by the README.md file when run without arguments

Run the script with --help to learn about the usage.

Dependencies:
- pycocotools
- numpy
- tqdm

"""

import os
import argparse

from pycocotools.coco import COCO
import numpy as np
import tqdm
import shutil
import json

def sample(annot_path, images_path, output_postfix="_small", percentage=0.01):
    print(f"Sampling {percentage:.2f}% from {annot_path} & {images_path} and saving in same dirs with postfix '{output_postfix}'")
    coco=COCO(annot_path)
    ids = coco.getImgIds()

    N = int(len(ids)*percentage)
    print(f"Choosing {N}/{len(ids)} images randomly.")
    ids = np.random.choice(ids, size=N)

    output_annot_path = annot_path.replace(".json", f"{output_postfix}.json")
    print(f"Generating sampled coco .json file {output_annot_path}.")
    result = dict(info=coco.dataset["info"], licenses=coco.dataset["licenses"],
                  categories=list(coco.cats.values()),
                  images=list(), annotations=list())

    for i in tqdm.tqdm(ids, total=N):
        img = coco.imgs[i]
        result["images"].append(img)
        for annot in coco.imgToAnns[i]:
            result["annotations"].append(annot)

    with open(output_annot_path, "w") as f:
        json.dump(result, f)

    output_images_path = os.path.join(os.path.dirname(images_path),
                                      os.path.basename(images_path)+output_postfix)
    print(f"Copying sampled images into sampled directory {output_images_path}")
    
    os.makedirs(output_images_path, exist_ok=True)
    for i in tqdm.tqdm(ids, total=N):
        shutil.copyfile(os.path.join(images_path, coco.imgs[i]["file_name"]),
                        os.path.join(output_images_path, coco.imgs[i]["file_name"]))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--annot_path", help="The path to the coco annotations json file", type=str, default=None)
    parser.add_argument("--images_path", help="The path to where the referenced images are", type=str, default=None)
    parser.add_argument("--output_postfix", type=str, default="_small")
    parser.add_argument("--percentage", type=float, default=0.01)
    
    args = parser.parse_args()

    if args.annot_path is None or args.images_path is None:
        print("annot_path or images_path have not been specified. " 
              "Using default behavior which assumes dataset is in default location.")
        sample("./dataset/livecell_coco_train.json", "./dataset/images/livecell_train_val_images",
               args.output_postfix, args.percentage)
        sample("./dataset/livecell_coco_val.json", "./dataset/images/livecell_train_val_images",
               args.output_postfix, args.percentage)
        sample("./dataset/livecell_coco_test.json", "./dataset/images/livecell_test_images",
               args.output_postfix, args.percentage)
    else:
        sample(args.annot_path, args.images_path, args.output_postfix, args.percentage)


