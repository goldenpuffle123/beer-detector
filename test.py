from ultralytics import YOLO
import cv2
import time


TARGET_CLASSES = ["beer"]
ALPHA = 0.3
model = YOLO("runs/detect/train4/weights/best.pt", verbose=False)  # trained

cap = cv2.VideoCapture(0)
no_beer = cv2.VideoCapture("beer_videos/no_beer.mp4")
yes_beer = cv2.VideoCapture("beer_videos/yes_beer.mp4")

beer_detected = False

def get_overlay_frame(overlay_vid: cv2.VideoCapture):
    ret, frame = overlay_vid.read()
    if not ret:
        overlay_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = overlay_vid.read()
    return frame

start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name in TARGET_CLASSES:
                beer_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
    if time.time() - start > 5:
        beer_detected = True
                
    if not beer_detected:
        overlay = get_overlay_frame(no_beer)

        # Resize overlay to webcam frame
        if overlay is not None:
            overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

            # Alpha blend
            frame = cv2.addWeighted(
                frame, 1 - ALPHA,
                overlay, ALPHA,
                0
            )

        cv2.putText(
            frame,
            "NO BEER DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:
        overlay = get_overlay_frame(yes_beer)

        # Resize overlay to webcam frame
        if overlay is not None:
            overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

            # Alpha blend
            frame = cv2.addWeighted(
                frame, 1 - ALPHA,
                overlay, ALPHA,
                0
            )

        cv2.putText(
            frame,
            "BEER DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Beer Can Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()