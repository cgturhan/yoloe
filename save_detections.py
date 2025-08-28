import json
import uuid
from pathlib import Path
import os
import hashlib
from PIL import Image

# Global containers
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

supercategory_map = {
    "person": "human",
    "pedestrian": "human",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "trailer": "vehicle",
    "traffic_light": "traffic_item",
    "traffic sign": "traffic_item",
    "construction_cone": "object",
    "barrier": "object",
    "dog": "animal",
    "bike":"vehichle",
    "tram":"vehicle",
    "train":"vehicle",
    "cyclist":"human",
    "road":"surface",
    "building":"background",
    "tree":"background"
}
default_supercategory = "object"

def generate_image_id_from_path(path: str) -> int:
    return int(hashlib.sha1(path.encode("utf-8")).hexdigest(), 16) % (10 ** 12)

def save_detections_to_cocoformat(detections, image_filename, class_names):
    """
    Save detections to a COCO format JSON file.

    Args:
        detections: supervision.Detections object.
        image_filename: Full path to the image.
        class_names: List of class names (indexed by class_id).
    """
    image_filename = str(image_filename)  # ensure it's a string
    image_id = generate_image_id_from_path(image_filename)
    input_image = Image.open(image_filename).convert("L") 
    image_height, image_width = input_image.size 

    # Parse path info
    file_path = Path(image_filename)
    file_folder = file_path.parent
    img_name = file_path.stem
    output_name = f"{img_name}-detections.json"
    output_path = file_folder / output_name

    # 1. Add image info
    coco_output["images"].append({
        "id": image_id,
        "file_name": file_path.name,
        "width": image_width,
        "height": image_height
    })

    # 2. Add category info (only once)
    if not coco_output["categories"]:  # avoid duplicates
        for i, class_name in enumerate(class_names):
            supercategory = supercategory_map.get(class_name, default_supercategory)
            coco_output["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": supercategory
            })

    # 3. Add annotation info
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        width = x2 - x1
        height = y2 - y1
        class_id = int(detections.class_id[i])
        confidence = float(detections.confidence[i])

        annotation = {
            "id": len(coco_output["annotations"]) + 1,  # globally unique
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [float(x1), float(y1), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": 0,
            "score": confidence  # optional (for predictions)
        }
        coco_output["annotations"].append(annotation)

    return coco_output

