import datetime
import random
import argparse
import shutil
import os
import json
import ipdb
import xml.etree.ElementTree as ET

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=".")
    parser.add_argument("-o", "--output_dir", default="results")
    parser.add_argument("--p_train", type=float, default=0.6)
    parser.add_argument("--p_val", type=float, default=0.2)
    parser.add_argument("--p_test", type=float, default=0.2)
    parser.add_argument("-t", "--timestamp")
    parser.add_argument("-a", "--author", default="example@example.com")
    args = parser.parse_args()


    if args.p_train + args.p_val + args.p_test < 0.995:
        exit("Dataset split proportions don't add to 1")

    if args.timestamp == None:
        timestamp = str(datetime.datetime.now())
    else:
        timestamp = args.timestamp

    tree = ET.parse(f"{args.input_dir}/annotations.xml")
    root = tree.getroot()
    
    classes = {}
    classes["classes"] = []

    meta = root.find("meta")
    tasks = meta.findall("task")
    
    class_list = set()

    # make necessary dirs for output files
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/data/", exist_ok=True)

    # Parse possible output classes and
    # write to output directory
    for task in tasks:
        labels = task.find("labels")
        for label in labels:
            class_list.add(label.find("name").text)

    for c in class_list:
        classes["classes"].append({ "id" : c })
    
    classes_json = json.dumps(classes, indent=4)
    with open(f"{args.output_dir}/classes.json", "w") as outfile:
        outfile.write(classes_json)

    # Parse each image for bounding boxes and write
    # to dataset output directory
    all_labels = []

    for image in root.findall("image"):
        objects_dict = {}
        objects_dict["objects"] = []
        
        width = float(image.get("width"))
        height = float(image.get("height"))

        name = image.get("name")
        idx = int(image.get("id"))
        id_str = f"{idx:06d}"
        all_labels.append(id_str)

        for box in image.findall("box"):
            label_dict = {}

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            label_dict["class_id"] = box.get("label")
            label_dict["x"] = (xtl + xbr) / width
            label_dict["y"] = (ytl + ybr) / height
            label_dict["h"] = (ybr - ytl) / height
            label_dict["w"] = (xbr - xtl) / width
        
            objects_dict["objects"].append(label_dict)

        shutil.copy(f"{args.input_dir}/images/{name}.PNG", f"{args.output_dir}/data/{id_str}.png")
        
        with open(f"{args.output_dir}/data/{id_str}.json", "w+") as f:
            objects_json = json.dumps(objects_dict, indent=4)
            f.write(objects_json)
    
    # parse and write metadata
    meta_dict = {}
    meta_dict["author"] = args.author
    meta_dict["date"] = timestamp
    meta_dict["type"] = "segmentation"
    meta_dict["md5"] = ""

    meta_json = json.dumps(meta_dict, indent=4)

    with open (f"{args.output_dir}/meta.json", "w") as f:
        f.write(meta_json)

    # generate random splits and write to json
    random.shuffle(all_labels)

    train_idx = int(len(all_labels) * args.p_train)
    val_idx = int(len(all_labels) * (args.p_train + args.p_val))

    splits_dict = {}
    splits_dict["splits"] = {}
    splits_dict["splits"]["train"] = sorted(all_labels[:train_idx])
    splits_dict["splits"]["val"] = sorted(all_labels[train_idx:val_idx])
    splits_dict["splits"]["test"] = sorted(all_labels[val_idx:])

    splits_json = json.dumps(splits_dict, indent=4) 

    with open(f"{args.output_dir}/splits.json", "w") as f:
        f.write(splits_json)
