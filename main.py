from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import Teamassigner
# from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from pose_detector import PoseDetector
from gait_analysis import GaitAnalysis
import sys
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('E:/Users/Public/ProjectICS/All_videos/002/5238-17_700000.avi')
    # video_frames = read_video('C:/Users/joseph/Downloads/Video/makena.mp4')

    # Initialize Tracker
    tracker = Tracker('models/bestly.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # # camera movement estimator
    # camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
    #                                                                             read_from_stub=True,
    #                                                                             stub_path='stubs/camera_movement_stub.pkl')

    # print(type(tracks["Runner"]))  # Should be <class 'list'>
    # print(len(tracks["Runner"]))  
    # Interpolate runner position
    # tracks["Runner"] = tracker.interpolate_runner_positions(tracks["Runner"])
    
    # sys.exit()  #stops execution

    # Assign Runner Teams
    # team_assigner = Teamassigner()

    # team_assigner.assign_team_color(video_frames[0],
    #                                 tracks['Runner'][0]
    #                                 )
    # for frame_num, runner_track in enumerate(tracks['Runner']):
    #     for runner_id, track in runner_track.items():
    #         if len(video_frames) == 0:
    #             print("No frames available to process. Exiting.")
    #         else:
                
    #             team = team_assigner.get_runner_team(video_frames[frame_num],
    #                                                 track['bbox'],
    #                                                 runner_id
    #                                                 )
    #             tracks['Runner'][frame_num][runner_id]['team'] = team
    #             tracks['Runner'][frame_num][runner_id]['team_color'] = team_assigner.team_colors[team]




    # # save cropped image
    # for track_id, runner in tracks['Runner'][0].items():
    #     bbox = runner['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save the cropped image
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
    #     break


    
    # Initialize PoseDetector
    pose_detector = PoseDetector()

    # Initialize GaitAnalysis with the pose_detector instance
    gait_analyzer = GaitAnalysis(pose_detector)

    # Process frames for Pose Detection
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):

        # tracks = tracker.interpolate_runner_positions_for_missing_track(tracks, frame_num)
        # Get the runner's bounding box for pose detection
        runner_tracks = tracks['Runner'][frame_num]
        
        # for runner_id, runner_info in runner_tracks.items():
        #     print(runner_info)
        #     sys.exit()  #stops execution

        #     bbox = runner_info['bbox']
        #     x1, y1, x2, y2 = map(int, bbox)

        #     # Crop the frame to the runner's bounding box
        #     cropped_frame = frame[y1:y2, x1:x2]

        #     # Detect pose for the cropped frame
        #     pose_results = pose_detector.detect_pose(cropped_frame)
            

        #            # Estimate height only if runner is upright
        #     if pose_results.pose_landmarks:
        #         is_upright = pose_detector.is_upright(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
                
        #         gait_analysis = gait_analyzer.analyze_gait(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

        #         # Display gait analysis on the frame
        #         if gait_analysis['left_stride']:
        #             cv2.putText(frame, f"Left Stride: {gait_analysis['left_stride']:.2f} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #         if gait_analysis['right_stride']:
        #             cv2.putText(frame, f"Right Stride: {gait_analysis['right_stride']:.2f} px", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #         cv2.putText(frame, gait_analysis['overstriding_warning'], (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #         # Optionally, display the hip angle
        #         cv2.putText(frame, f"Hip Angle: {gait_analysis['hip_angle']:.2f}", (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        #         if is_upright:
        #             # Estimate the runner's height using pose landmarks
        #             runner_height = pose_detector.estimate_runner_height(frame, pose_results, (x1, y1, x2, y2))

        #             # Refine the gradient calculation using the hips and check foot contact
        #             slope, angle, surface_type = pose_detector.check_foot_contact(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

        #             # Interpret the surface gradient based on the angle
        #             if surface_type:
        #                 surface_type = pose_detector.interpret_gradient(angle)
                    
        #             # Draw the estimated height of the runner on the frame
        #             if runner_height:
        #                 cv2.putText(
        #                     frame, 
        #                     f"height in meters: {runner_height:.2f} m",  # Text to display
        #                     (x1, y1 - 10),  # Position above the bounding box
        #                     cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        #                     1,  # Font size (scale)
        #                     (0, 255, 0),  # Font color (green)
        #                     2,  # Line thickness
        #                     cv2.LINE_AA  # Anti-aliased line
        #                 )

        #                 # Optionally, display the angle on the frame
        #                 if surface_type:
        #                     cv2.putText(frame, f"Surface Gradient: {surface_type}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    
            

         
        #     # Draw pose landmarks and connections on the original frame
        #     frame = pose_detector.draw_pose(frame, pose_results, (x1, y1, x2, y2))

        
        if isinstance(runner_tracks, list):
            
            for runner_info in runner_tracks:
                # print(runner_info)
                # sys.exit()  #stops execution

                bbox = runner_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Crop the frame to the runner's bounding box
                cropped_frame = frame[y1:y2, x1:x2]

                # Detect pose for the cropped frame
                pose_results = pose_detector.detect_pose(cropped_frame)
                

                    # Estimate height only if runner is upright
                if pose_results.pose_landmarks:
                    is_upright = pose_detector.is_upright(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
                    
                    gait_analysis = gait_analyzer.analyze_gait(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

                    # Display gait analysis on the frame
                    if gait_analysis['left_stride']:
                        cv2.putText(frame, f"Left Stride: {gait_analysis['left_stride']:.2f} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if gait_analysis['right_stride']:
                        cv2.putText(frame, f"Right Stride: {gait_analysis['right_stride']:.2f} px", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, gait_analysis['overstriding_warning'], (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    

                    if is_upright:
                        
                        # Refine the gradient calculation using the hips and check foot contact
                        slope, angle, surface_type = pose_detector.check_foot_contact(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

                        # Interpret the surface gradient based on the angle
                        if surface_type:
                            surface_type = pose_detector.interpret_gradient(angle)
                        
                        

                # Draw pose landmarks and connections on the original frame
                frame = pose_detector.draw_pose(frame, pose_results, (x1, y1, x2, y2))
                frame = pose_detector.draw_pose_with_angles(frame, pose_results, (x1, y1, x2, y2))
       
        
        else:

            for runner_id, runner_info in runner_tracks.items():
                # print(runner_info)
                # sys.exit()  #stops execution

                bbox = runner_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Crop the frame to the runner's bounding box
                cropped_frame = frame[y1:y2, x1:x2]

                # Detect pose for the cropped frame
                pose_results = pose_detector.detect_pose(cropped_frame)
                

                    # Estimate height only if runner is upright
                if pose_results.pose_landmarks:
                    is_upright = pose_detector.is_upright(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
                    
                    gait_analysis = gait_analyzer.analyze_gait(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

                    # Display gait analysis on the frame
                    if gait_analysis['left_stride']:
                        cv2.putText(frame, f"Left Stride: {gait_analysis['left_stride']:.2f} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if gait_analysis['right_stride']:
                        cv2.putText(frame, f"Right Stride: {gait_analysis['right_stride']:.2f} px", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, gait_analysis['overstriding_warning'], (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    
                    if is_upright:
                       
                        # Refine the gradient calculation using the hips and check foot contact
                        slope, angle, surface_type = pose_detector.check_foot_contact(pose_results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])

                        # Interpret the surface gradient based on the angle
                        if surface_type:
                            surface_type = pose_detector.interpret_gradient(angle)
                        
                        
                # Draw pose landmarks and connections on the original frame
                frame = pose_detector.draw_pose(frame, pose_results, (x1, y1, x2, y2))
                frame = pose_detector.draw_pose_with_angles(frame, pose_results, (x1, y1, x2, y2))

       
                

            
                
            
        
        # Optionally, draw camera movement on the frame
    
        # Draw object Tracks
        output_video_frames = tracker.draw_annotations(video_frames, tracks)
        # Draw camera movement
        # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        
        output_video_frames.append(frame)


    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()