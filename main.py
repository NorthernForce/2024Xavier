from math import atan

from ultralytics import YOLO
import cv2
import ntcore
from cscore import CameraServer

model = YOLO("YOLOv8nNO.pt").to("cuda")

video = cv2.VideoCapture(0)
#video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not video.isOpened():
    raise RuntimeError(f"Could not open a webcam using {video.getBackendName()}")

instance = ntcore.NetworkTableInstance.getDefault()

camera_out = CameraServer.putVideo("Note Detection", 640, 480)

table = instance.getTable("note_detection")
note_rad = table.getFloatTopic("note_rad").publish()

instance.setServerTeam(172)
instance.startClient4("skynet")

try:
    while True:
        ret, frame = video.read()
        
        results = model(frame, conf=.45, verbose=False)[0]

        largest_conf = 0
        current_note_rad = float("nan") # NaN = no note detected
        current_note_x = 0
        current_note_y = 0
        for box in results.boxes:
            x, y, w, h = box.xywh[0]
            # inverse tangent of (note_x - principal_point_x) / focal_len_x
            #x_rad = atan((x.item() - 323.4001746867161) / 473.31513614924415)
            #x_rad = atan((x.item() - 260.5461196441517) / 326.2388384431502) #for 640x480 cropped
            #x_rad = atan((x.item() - 326.2388384431502) / 260.5461196441517)
            #x_rad = atan((x.item() - 390.8191794662276) / 652.4776768863004)
            x_rad = atan((x.item() - 652.4776768863004) / 390.8191794662276)

            cv2.drawMarker(
                frame,
                (int(x.item()), int(y.item())),
                (0, 255, 0), cv2.MARKER_CROSS, 30, 8)
            cv2.rectangle(frame,
                (int(x-w//2), int(y-h//2)),
                (int(x+w//2), int(y+h//2)), (0, 255, 0), 8)
            if box.conf.item() > largest_conf:
                largest_conf = box.conf.item()
                current_note_rad = x_rad
                current_note_x = int(x.item())
                current_note_y = int(y.item())
        note_rad.set(current_note_rad)
        cv2.drawMarker(
            frame,
            (current_note_x, current_note_y),
            (255, 0, 0), cv2.MARKER_CROSS, 30, 8)
        camera_out.putFrame(frame)
except KeyboardInterrupt:
    print("Closing...")
    video.release()

print("Closed.")
