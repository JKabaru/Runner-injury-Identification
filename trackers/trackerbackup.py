from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


class Tracker1:    #remember you changed this
    def __init__(self, model_path):
        self.model= YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_runner_positions(self, tracks):
        runner_positions = []

        # Collect runner positions (already done)
        for frame_data in tracks:
            if isinstance(frame_data, dict):
                for key, value in frame_data.items():
                    if 'bbox' in value:
                        runner_positions.append(value['bbox'])

        # print("Collected runner positions:", runner_positions)

        # We assume runner_positions is a list of [x1, y1, x2, y2] for each frame
        if len(runner_positions) < 2:
            # print("Not enough data to interpolate.")
            return runner_positions  # Not enough data to interpolate

        # Extract individual components of the bounding boxes (x1, y1, x2, y2)
        x1 = np.array([bbox[0] for bbox in runner_positions])
        y1 = np.array([bbox[1] for bbox in runner_positions])
        x2 = np.array([bbox[2] for bbox in runner_positions])
        y2 = np.array([bbox[3] for bbox in runner_positions])

        # Define a time axis (frame numbers)
        frames = np.arange(len(runner_positions))

        # Create a cubic spline for each coordinate (x1, y1, x2, y2)
        spline_x1 = CubicSpline(frames, x1)
        spline_y1 = CubicSpline(frames, y1)
        spline_x2 = CubicSpline(frames, x2)
        spline_y2 = CubicSpline(frames, y2)

        # Interpolate between frames (for example, interpolate to create 5 frames between each existing frame)
        interpolated_positions = []
        frame_count = len(runner_positions)
        
        for i in range(frame_count - 1):
            # Generate 5 interpolated frames between frames[i] and frames[i+1]
            for t in np.linspace(i, i + 1, 6)[1:-1]:  # Skip t=0 and t=1 to avoid duplicates
                interpolated_bbox = [
                    spline_x1(t),  # Interpolated x1
                    spline_y1(t),  # Interpolated y1
                    spline_x2(t),  # Interpolated x2
                    spline_y2(t)   # Interpolated y2
                ]
                interpolated_positions.append(interpolated_bbox)

        # Add the original bounding boxes at the start and end
        interpolated_positions = runner_positions[:1] + interpolated_positions + runner_positions[-1:]

        # Print the interpolated positions to verify
        # print("Interpolated runner positions:", interpolated_positions)

        return interpolated_positions
        
        
    def detect_frames(self, frames):
        batch_size=30
        detections = []
        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.6)  
            detections += detections_batch
        return detections  

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)


        tracks={
            "Runner" : []    # 
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # convert to supervision detection format 
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # tracks["Runner"].append({})

            if frame_num not in tracks["Runner"]:
                 tracks["Runner"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['Runner']:
                    tracks["Runner"][frame_num][track_id] = {"bbox": bbox}

            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:  
                pickle.dump(tracks,f)  

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
         y2 = int(bbox[3])

         x_center, _ = get_center_of_bbox(bbox)
         width = get_bbox_width(bbox)

         cv2.ellipse(
             frame,
             center = (x_center, y2),
             axes = (int(width), int(0.35*width)), 
             angle = 0.0,
             startAngle=45,
             endAngle=235,
             color= color,
             thickness= 2,
             lineType= cv2.LINE_4

         )

         rectangle_width = 40
         rectanlge_height = 20
         x1_rect = x_center - rectangle_width/2
         x2_rect = x_center + rectangle_width/2
         y1_rect = (y2 - rectanlge_height/2) +15
         y2_rect = (y2 + rectanlge_height/2) +15

         if track_id is not None:
             cv2.rectangle(frame,
                           (int(x1_rect), int(y1_rect) ),
                           (int(x2_rect), int(y2_rect) ),
                           color,
                           cv2.FILLED
                           
                           )
             
             X1_text = x1_rect+12
             if track_id > 90:
                 X1_text -=10

             cv2.putText(
                 frame,
                 f"{track_id}",
                 (int(X1_text), int(y1_rect + 15)),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 0.6,
                 (0,0,0),
                 2
             )
         

         return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            runner_dict = tracks["Runner"][frame_num]
            
            # Draw for runner 

            for track_id, runner in runner_dict.items():
                color = runner.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, runner["bbox"], color, track_id)

            output_video_frames.append(frame)

        return output_video_frames