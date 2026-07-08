import os
import sys
import json
import cv2
import numpy as np
import skimage.io
from pathlib import Path

# Add mrcnn to path if needed (installed via pip usually places it correctly)
import mrcnn.model as modellib
from mrcnn.config import Config

class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + (young and senescent)
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

def extract_polygons(mask):
    """
    Extract polygons from a boolean mask.
    mask: [H, W] boolean or uint8 mask
    Returns: list of dicts with all_points_x, all_points_y
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Ignore tiny artifacts
        if len(contour) > 5:
            contour = contour.squeeze(1) # shape (N, 2)
            # OpenCV contours are (x, y)
            poly = {
                "name": "polygon",
                "all_points_x": [int(x) for x in contour[:, 0]],
                "all_points_y": [int(y) for y in contour[:, 1]]
            }
            polygons.append(poly)
    return polygons

def main(weights_path, images_dir, output_json_path):
    config = InferenceConfig()
    config.display()

    print("Loading model in inference mode...")
    # By default, tf.device('/gpu:0') or '/cpu:0'
    model = modellib.MaskRCNN(mode="inference", model_dir="logs", config=config)
    
    print(f"Loading weights from {weights_path}")
    model.keras_model.load_weights(weights_path, by_name=True)
    
    class_names = ['BG', 'young', 'senescent']

    via_dict = {}

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process.")

    for filename in image_files:
        image_path = os.path.join(images_dir, filename)
        try:
            image = skimage.io.imread(image_path)
            # Ensure RGB
            if image.ndim == 2:
                image = skimage.color.gray2rgb(image)
            elif image.shape[-1] == 4:
                image = image[..., :3]
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
            
        file_size = os.path.getsize(image_path)
        via_key = f"{filename}{file_size}"
        
        print(f"Processing {filename} ...")
        results = model.detect([image], verbose=0)
        r = results[0]

        regions = []
        num_instances = r['rois'].shape[0]
        
        for i in range(num_instances):
            class_id = r['class_ids'][i]
            score = r['scores'][i]
            mask = r['masks'][:, :, i]
            
            label_str = class_names[class_id]
            
            polygons = extract_polygons(mask)
            for poly in polygons:
                regions.append({
                    "shape_attributes": poly,
                    "region_attributes": {
                        "cell": label_str,
                        "score": float(score)  # keeping score for reference, VIA ignores unknown attributes
                    }
                })
                
        via_dict[via_key] = {
            "filename": filename,
            "size": file_size,
            "regions": regions,
            "file_attributes": {}
        }
        
    print(f"Saving annotations to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(via_dict, f)

    print("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Auto-annotate images using Mask R-CNN")
    parser.add_argument('--weights', type=str, required=True, help="Path to .h5 weights file")
    parser.add_argument('--images', type=str, required=True, help="Directory containing images to annotate")
    parser.add_argument('--output', type=str, required=True, help="Path to save via_project.json")
    
    args = parser.parse_args()
    main(args.weights, args.images, args.output)
