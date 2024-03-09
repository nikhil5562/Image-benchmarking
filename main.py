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
    source, imgsz, nosave, csv_path = opt.source, opt.img_size, opt.no_save, opt.csv_path
    
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
        
        if not image_files:
            print(f"No supported image files found in {emotion_path}")
            continue
        
        # Set DataLoader for images in the emotion folder
        dataset = LoadImages(emotion_path, img_size=imgsz, stride=stride)
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Run once
        
        for path, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # Uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            pred = model(img, augment=opt.augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=opt.agnostic_nms)
            
            # Process detections
            for i, det in enumerate(pred):  # Detections per image
                p, im0 = path, im0s
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    images = []
                    for *xyxy, _, _ in reversed(det):
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        images.append(im0[int(y1):int(y2), int(x1):int(x2)])
                    
                    if images:
                        emotions = detect_emotion(images, show_conf=False)
                        emotion_percentages = emotions[0]
                        dominant_emotion = max(emotion_percentages, key=lambda x: float(x[1][:-1]))[0]
                        
                        # Write results to CSV file
                        image_name = os.path.basename(p)
                        row = f"{emotion_folder},{image_name},"
                        row += ",".join([f"{percentage[1]}" for percentage in emotion_percentages])
                        row += f",{dominant_emotion}\n"
                        with open(csv_path, 'a') as csv_file:
                            csv_file.write(row)
                        
                        # Save results (image with detections)
                        if not nosave:
                            output_path = os.path.join(emotion_folder, image_name)
                            cv2.imwrite(output_path, im0)

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
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt=opt)