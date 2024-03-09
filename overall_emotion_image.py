from pathlib import Path
import numpy as np
import argparse
import time
import os

import torch.backends.cudnn as cudnn
import torch
import cv2

from emotion import detect_emotion, init

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(opt):
    source, imgsz, nosave, csv_path, output_folder = opt.source, opt.img_size, opt.no_save, opt.csv_path, opt.output
    
    # Initialize
    set_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init(device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA
    
    # Load model
    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size
    if half:
        model.half()  # To FP16
    
    # Get names and colors
    names = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    # Create CSV file and write header
    with open(csv_path, 'w') as csv_file:
        csv_file.write("label,image_name,anger,contempt,disgust,fear,happy,neutral,sad,surprise,dominant_emotion\n")
    
    # Process images in each emotion folder
    for emotion_folder in names:
        emotion_path = os.path.join(source, emotion_folder)
        if not os.path.exists(emotion_path):
            print(f"Folder not found: {emotion_path}")
            continue
        
        # Get list of image files in the emotion folder
        image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'))]
        
        print(f"Files in {emotion_path}: {os.listdir(emotion_path)}")
        
        if not image_files:
            print(f"No supported image files found in {emotion_path}")
            continue
        
        # Process each image file
        for image_file in image_files:
            image_path = os.path.join(emotion_path, image_file)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (imgsz, imgsz))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_expanded = np.expand_dims(img_transposed, axis=0)
            img_tensor = torch.from_numpy(img_expanded).to(device)
            img_tensor = img_tensor.half() if half else img_tensor.float()  # Uint8 to fp16/32
            
            # Inference
            pred = model(img_tensor, augment=opt.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=opt.agnostic_nms)
            
            # Process detections
            for i, det in enumerate(pred):  # Detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
                    
                    images = []
                    for *xyxy, _, _ in reversed(det):
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                        cropped_img_resized = cv2.resize(cropped_img, (224, 224))
                        cropped_img_normalized = cropped_img_resized.astype(np.float32) / 255.0
                        cropped_img_transposed = np.transpose(cropped_img_normalized, (2, 0, 1))
                        images.append(cropped_img_transposed)
                    
                    if images:
                        emotions = detect_emotion(images)
                        emotion_percentages = emotions[0]
                        dominant_emotion = max(emotion_percentages, key=lambda x: float(x[1][:-1]))[0]
                        
                        # Write results to CSV file
                        image_name = os.path.basename(image_path)
                        row = f"{emotion_folder},{image_name},"
                        row += ",".join([f"{percentage[1]}" for percentage in emotion_percentages])
                        row += f",{dominant_emotion}\n"
                        with open(csv_path, 'a') as csv_file:
                            csv_file.write(row)
                        
                        # Draw bounding box and label on the image
                        label = f"{dominant_emotion}"
                        color = colors[names.index(dominant_emotion)]
                        plot_one_box(xyxy, img, label=label, color=color, line_thickness=2)
                        
                        # Save results (image with detections)
                        if not nosave:
                            output_path = os.path.join(output_folder, emotion_folder, image_name)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no-save', action='store_true', help='do not save images')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--csv-path', default="emotion_results.csv", help='path to save the emotion results CSV file')
    parser.add_argument('--output', default="output", help='output folder to save processed images')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt=opt)