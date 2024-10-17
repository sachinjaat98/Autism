import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter


#-------------------project starts here-------------------#
model = YOLO('autism_modelm.pt')

#--------- Initialize tracking variables-----------#
trackers = {}
next_id = 0
colors = {'child': (0, 255, 0), 'therapist': (255, 0, 0)}  # Green for children, Red for therapists
track_colors = {}  # To store assigned colors for each track ID


#####Function to initialize Kalman filter###############
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])  # initial state (x, y, vx, vy)
    kf.P *= 1000.0  # state covariance
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # state transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # measurement function
    return kf

# Open video file
video_path = 'Matching.mp4'
output_video_path = 'output4.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Extract detected boxes
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf >= 0.7:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = model.names[int(box.cls[0])]
                detected_boxes.append((label, (int(x1), int(y1), int(x2), int(y2)))) 

    # Update trackers
    tracked_ids = set()  # To keep track of which IDs are currently tracked
    for label, (x1, y1, x2, y2) in detected_boxes:
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        tracked = False

        # Check if the person is already being tracked
        for track_id in list(trackers.keys()):
            kf = trackers[track_id]
            kf.predict()
            predicted_position = (int(kf.x[0]), int(kf.x[1]))

            # Check distance from predicted position to current centroid
            if np.linalg.norm(np.array(predicted_position) - np.array(centroid)) < 50:  # Threshold distance
                kf.update(np.array([centroid[0], centroid[1]]))
                tracked_ids.add(track_id)
                tracked = True
                break

        if not tracked:  # If not tracked, create a new tracker
            kf = initialize_kalman_filter()
            kf.x = np.array([centroid[0], centroid[1], 0, 0])
            trackers[next_id] = kf
            track_colors[next_id] = colors.get(label, (255, 255, 255))  # Assign color
            next_id += 1

    # Remove stale trackers
    for track_id in list(trackers.keys()):
        kf = trackers[track_id]
        kf.predict()
        x, y = int(kf.x[0]), int(kf.x[1])

        # Check if the predicted position is still within the frame
        if track_id not in tracked_ids:
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                del trackers[track_id]  # Remove stale tracker if it's out of frame

    # Draw tracked objects
    for track_id, kf in trackers.items():
        kf.predict()
        x, y = int(kf.x[0]), int(kf.x[1])

        # Use the pre-assigned color
        color = track_colors.get(track_id, (255, 255, 255))  # Default to white if not found

        # Draw bounding box and ID
        for label, (det_x1, det_y1, det_x2, det_y2) in detected_boxes:
            if (det_x1 <= x <= det_x2) and (det_y1 <= y <= det_y2):
                cv2.rectangle(frame, (det_x1, det_y1), (det_x2, det_y2), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (det_x1, det_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break  # Exit after drawing to avoid drawing multiple boxes for one tracker

    # Save the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
