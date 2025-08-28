import argparse
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from ultralytics import YOLOE
import supervision as sv

from save_detections import save_detections_to_cocoformat



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the data folder"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path or ID of the model checkpoint"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic_light", "traffic sign", "train", "tram", "road"],
        help="List of class names to set for the model"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to save the annotated image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
    )
    # New argument for confidence threshold
    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help="Minimum confidence threshold to keep detections"
    )     
    parser.add_argument(
        "--remove_overlaps",
        type=bool,
        default=True,
        help="Whether to remove the overlapped detections"
    )
    parser.add_argument(
        "--return_road_mask",
        action="store_true",
        help="Whether to filter the other masks"
    )
    parser.add_argument(
        "--return_instance_mask",
        action="store_true",
        help="Whether to filter the other masks"
    )
    parser.add_argument(
        "--return_detection",
        action="store_true",
        help="Whether to filter the other masks"
    )
    parser.add_argument(
        "--save_annotations",
        type=bool,
        default=True,
        help="Whether to filter the other masks"
    )
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help = "Whether to return labels with the annotated image"
    )
    parser.add_argument(
        "--priority_cls",
        type=str,
        default ="tram",
        help = "Whether to give priority to a class"
    ) 

def chunked(iterable, batch_size):
    """Yield successive batches from an iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def main():
    args = parse_args()

    # Load model once
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.eval()
    model.set_classes(args.names, model.get_text_pe(args.names))

    # Collect image paths
    image_paths = sorted([p for p in Path(args.source).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    batch_size = 8  # 👈 set your batch size here

    # Loop through image batches
    for batch_paths in chunked(image_paths, batch_size):
        # Load current batch of images
        batch_images = [Image.open(str(p)).convert("RGB") for p in batch_paths]

        # Run batch prediction
        results = model.predict(batch_images, verbose=False)  # 🔁 returns list of results

        # Process each image-result pair
        for image_path, image, result in zip(batch_paths, batch_images, results):
            file_path = Path(image_path)
            file_folder = file_path.parent
            out_name = file_path.stem
            ext = image_path.split('.')[-1]
            detections = sv.Detections.from_ultralytics(result)
                # Filter by confidence
            conf_mask = detections.confidence > args.conf
            detections = detections[conf_mask]

            if args.priority_cls!=None and len(detections) > 0:
                all_boxes = torch.tensor(np.array(detections.xyxy), dtype=torch.float32)
                iou_threshold = 0.65
                if args.priority_cls in detections["class_name"]:              
                    filtered_indices = [i for i, cls in enumerate(detections["class_name"]) if cls == args.priority_cls]

                    # If no trams, keep everything
                    if not filtered_indices:
                        keep_mask = torch.ones(len(detections), dtype=torch.bool)

                    else:
                        # Convert all boxes to tensor                
                        filtered_boxes = all_boxes[filtered_indices]
                        iou_matrix = box_iou(filtered_boxes, all_boxes) 

                        # Mask of detections to remove (overlapping with tram)
                        remove_mask = (iou_matrix > iou_threshold).any(dim=0)  # shape: (num_detections,)
                        remove_mask[filtered_indices] = False
                        keep_mask = ~remove_mask

                        detections = sv.Detections(
                            xyxy=detections.xyxy[keep_mask],
                            confidence=detections.confidence[keep_mask],
                            class_id=detections.class_id[keep_mask],
                            tracker_id=detections.tracker_id[keep_mask] if detections.tracker_id is not None else None,
                            mask=detections.mask[keep_mask] if detections.mask is not None else None,
                        )          
                    
            # Prepare labels
            class_names = [args.names[class_id] for class_id in detections.class_id.tolist()]
            if args.show_labels:
                labels = []
                for i, (class_name, confidence) in enumerate(zip(class_names, detections.confidence)):
                    label = f"{class_name} {confidence:.2f}"
                    labels.append(label)
        
            # Annotate
            resolution_wh = image.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

            annotated_image = image.copy()
            if args.return_road_mask:
                mask_cls = "road"
                if mask_cls in class_names:
                    priority_mask = torch.tensor([cls == mask_cls for cls in class_names])
                    filtered_detections = detections[priority_mask]

                    annotated_image = sv.MaskAnnotator(
                        color_lookup=sv.ColorLookup.CLASS,
                        opacity=0.4
                    ).annotate(scene=annotated_image, detections=filtered_detections)
        
                    #annotated_image = sv.BoxAnnotator(
                    #    color_lookup=sv.ColorLookup.CLASS,
                    #    thickness=thickness
                    #).annotate(scene=annotated_image, detections=detections[~priority_mask])
                else:
                    print("Road could not detected on given image!")
            if args.save_annotations:
                if args.return_instance_mask:
                    annotated_image = sv.MaskAnnotator(
                        color_lookup=sv.ColorLookup.CLASS,
                        opacity=0.4
                    ).annotate(scene=annotated_image, detections=detections)
                    output_file = f"{out_name}-masked{ext}"
               
                if args.show_labels:
                    annotated_image = sv.BoxAnnotator(
                        color_lookup=sv.ColorLookup.CLASS,
                        thickness=thickness
                    ).annotate(scene=annotated_image, detections=detections)
                    annotated_image = sv.LabelAnnotator(
                        color_lookup=sv.ColorLookup.CLASS,
                        text_scale=text_scale,
                        smart_position=True
                    ).annotate(scene=annotated_image, detections=detections, labels=labels)
                    output_file = f"{out_name}-annotated{ext}"
                
                annotated_image.save(os.path.join(args.output_folder, file_folder, output_file))
                print(f"Annotated image saved to: {output_file}")

            if args.return_detection:
                coco_output = save_detections_to_cocoformat(detections, args.source, args.names)
                with open(output_file, "w") as f:
                    json.dump(coco_output, f, indent=4)
                print(f"Saved detections in COCO format to: {output_file}")
   

if __name__ == "__main__":
    main()



   