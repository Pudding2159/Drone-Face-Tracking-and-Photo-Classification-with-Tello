import cv2
import torch
import threading
import torchvision.transforms as transforms
from djitellopy import Tello
from torchvision import models, datasets
from PIL import Image
from queue import Queue

def overlay_boxes_and_labels(frame, output, threshold=0.5):
    boxes = output['boxes'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score > threshold:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{MNIST_CLASSES[label]}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

MNIST_CLASSES = list(range(10))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

tello = Tello()
tello.connect()
tello.streamon()

frame_queue = Queue(maxsize=5)
processed_frame_queue = Queue(maxsize=1)

def detect_objects(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        return output[0]

def capture_frames():
    while True:
        frame = tello.get_frame_read().frame
        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.resize(frame, (360, 240))
            output = detect_objects(frame)
            overlay_boxes_and_labels(frame, output)
            if not processed_frame_queue.full():
                processed_frame_queue.put(frame)

capture_thread = threading.Thread(target=capture_frames)
processing_thread = threading.Thread(target=process_frames)
capture_thread.start()
processing_thread.start()

while True:
    if not processed_frame_queue.empty():
        frame = processed_frame_queue.get()
        cv2.imshow("Tello Live Video with Digit Detection", frame)

        key = cv2.waitKey(1) & 0xFF



        if key == ord("q"):
            break

tello.streamoff()
capture_thread.join()
processing_thread.join()
cv2.destroyAllWindows()