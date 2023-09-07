import cv2
import time
from PIL import Image
from torchvision import models, transforms
import torch
import json
import requests
import matplotlib.pyplot as plt
from Tello import *
import os

coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Replace the existing imagenet_labels variable with the coco_labels list
imagenet_labels = coco_labels

object_detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()





def save_1080p_screenshot(image, output_path):
    resized_image = cv2.resize(image, (1920, 1080))
    cv2.imwrite(output_path, resized_image)

output_dir = "screenshots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def detect_objects_and_draw_boxes(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = object_detection_model(input_tensor)

    boxes = output[0]['boxes'].cpu().numpy().tolist()
    labels = output[0]['labels'].cpu().numpy().tolist()
    scores = output[0]['scores'].cpu().numpy().tolist()

    annotated_image = frame.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            text = f'{imagenet_labels[label - 1]}: {score:.2%}'
            cv2.putText(annotated_image, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_image

w, h = 1280, 780
pid = [0.09, 0.09, 0]
pError = 0
startCounter = 0  # 0 - fly 1 - no fly

myDrone = initializeTello()

plt.ion()
while True:

    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1

    img = Tello_get_frame(myDrone, w, h)
    img, info = FindFace(img)
    pError = trackFace(myDrone, info, w, pid, pError)
    print(info[0][0])

    cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        annotated_image = detect_objects_and_draw_boxes(img)
        plt.clf()
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.draw()
        plt.pause(0.001)

        # Save the annotated image as a 1080p screenshot
        screenshot_path = os.path.join(output_dir, f'screenshot_{time.strftime("%Y%m%d-%H%M%S")}.jpg')
        save_1080p_screenshot(annotated_image, screenshot_path)

    if key == ord('q'):
        myDrone.land()
        break

plt.ioff()