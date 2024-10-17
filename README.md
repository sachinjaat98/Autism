# Therapist and Child Detection and Tracking

## Project Overview

This project implements a system for detecting and tracking children with Autism Spectrum Disorder (ASD) and therapists during therapy sessions. The goal is to assign unique IDs to individuals, track them throughout the video, handle re-entries into the frame, and maintain accurate ID assignment even after occlusions or partial visibility.

The system is designed to work on test videos provided, detecting children and therapists, assigning unique IDs to each, and tracking them as they move within the frame.

### Key Features:
1. **Person Detection**: Detects children and therapists using a trained YOLO model.
2. **Unique ID Assignment**: Assigns unique IDs to each person detected and ensures they remain constant throughout the video.
3. **Track Re-entries**: Tracks persons who leave and re-enter the frame, re-assigning the correct ID when they reappear.
4. **Post-Occlusion Tracking**: Maintains ID assignment even after occlusions or partial visibility.
5. **Save Predictions**: The system outputs the predictions and tracking results as an annotated video, where each detected person has their unique ID and bounding box overlaid.

---

## Setup Instructions

### Prerequisites:
- Python 3.7+
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- FilterPy (for Kalman filter)
- pytorch
- roboflow/labelimg


### Python Dependencies:
Install all necessary dependencies using bash
- pip install -r requirements.txt

requirements.txt includes:

- opencv-python
- numpy
- ultralytics
- filterpy
- torch
- torchvision
- torchaudio


## model selection
Download the pretrained trained YOLO model mainly yolov8 model through ultralytics library.

In this project I fine-tuned the pretrained yolov8 model using custom dataset and generated a autism_model, included in files.since it was very difficult to classify between child and therapist with pretrained models. so I extracted the images through image scrapper tools from web and then annotate dataset using roboflow.
then Increase the dataset using some image enhancement techniques with data augumentation technique , finaly genrated a dataset of nearly 1000 images.
then I finetune the yolov8  model with additional layers.

## Video Files:
- Test video files should be placed in the project directory. The model processes videos in .mp4 format, and the final output video will also be saved as an .mp4 file.

Running the Project:
Place the test video file in the project directory.

The output video with predictions and tracking will be saved as output_video_with_tracking.mp4.


## Inference Pipeline Details
1. Detection:
The detection is performed using a fine-tuned YOLO model (autism_modelm.pt). The model is trained on custom dataset consist of 970 images to detect:

- Children (ASD patients): Represented with a green bounding box.
- Therapists: Represented with a blue bounding box.
- Each detection is passed through a confidence threshold of 0.7 to ensure accurate predictions.

2. Tracking and ID Assignment:
Tracking is performed using the Kalman Filter, which helps in predicting the next position of each person based on their previous movements. Key aspects include:

Unique IDs: Each person detected is assigned a unique ID, which remains constant across frames.
Re-entry Tracking: When a person exits and re-enters the frame, their previous ID is reassigned based on proximity and tracking consistency.
Post-Occlusion: The tracker is robust enough to maintain the same ID after temporary occlusions or partial visibility.

3. Saving Predictions:
The bounding boxes, labels, and unique IDs are overlaid on each video frame. The processed frames are saved as a video file (output_video_with_tracking.mp4) for evaluation.

## Code Structure
- trackingv4.py: Main inference pipeline that performs detection, tracking, and saving the output video.
- autism_modelm.pt: YOLO model for detecting children and therapists.
- requirements.txt: Python dependencies for running the project.

## Expected Output
The output video shows:

Bounding Boxes: Overlaid around each person detected in the video (green for children, blue for therapists).
Unique IDs: Displayed above the bounding boxes, remaining constant throughout the video.
An example of the output video is stored as output_video_with_tracking.mp4.

Notes
Ensure that the model and test videos are placed in the correct directories.
The program outputs a video with predictions and tracking, which can be visualized with any video player supporting .mp4 format.


## link included
-test outputs
- different trained models 
- readme.md file
-requirement.txt

### Test Videos:
Download the test videos from the following link: https://drive.google.com/file/d/1VrDx8yF84GCvVAt8DAgBfA7e_814061Q/view?usp=sharing

