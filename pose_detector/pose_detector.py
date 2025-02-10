import cv2
import mediapipe as mp
import numpy as np
import math

class PoseDetector:  #remember you changed this 
    def __init__(self, static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Trackers for foot contact states
        self.left_foot_grounded = False
        self.right_foot_grounded = False
        self.left_foot_position = None
        self.right_foot_position = None

    def detect_pose(self, frame):
        # Convert the image to RGB as required by Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def draw_pose(self, frame, results, bbox):
        # Draw pose landmarks and connections on the frame
        if results.pose_landmarks:
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = results.pose_landmarks.landmark[start_idx]
                end_landmark = results.pose_landmarks.landmark[end_idx]

                # Transform normalized coordinates back to the original frame
                start_x = int(start_landmark.x * width) + x1
                start_y = int(start_landmark.y * height) + y1
                end_x = int(end_landmark.x * width) + x1
                end_y = int(end_landmark.y * height) + y1

                # Draw the connection on the frame
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            # Also draw the landmarks
            for landmark in results.pose_landmarks.landmark:
                landmark_x = int(landmark.x * width) + x1
                landmark_y = int(landmark.y * height) + y1
                cv2.circle(frame, (landmark_x, landmark_y), 5, (255, 0, 0), -1)

        return frame

    def interpolate_missing_landmarks(self, landmarks, required_points):
        """
        Interpolates missing keypoints using nearby visible keypoints.
        """
        interpolated_points = []
        for idx in required_points:
            if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                # Use the detected keypoint
                interpolated_points.append((landmarks[idx].x, landmarks[idx].y))
            else:
                # Handle missing keypoints
                if interpolated_points:  # Use average of previous points
                    avg_x = np.mean([p[0] for p in interpolated_points])
                    avg_y = np.mean([p[1] for p in interpolated_points])
                    interpolated_points.append((avg_x, avg_y))
                else:
                    interpolated_points.append((0, 0))  # Default if no data
        return interpolated_points
    
    def calculate_distance(self, point_a, point_b):
        """
        Calculate the Euclidean distance between two points.
        """
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def calculate_height(self, landmarks, width, height):
        """
        Calculate height based on head length using anthropometric proportions.
        """
        head_top = landmarks[0]  # Top of the head
        chin = landmarks[1]  # Chin

        head_length_pixels = abs(head_top[1] - chin[1]) * height

        # Using head length to estimate total height
        total_height_pixels = head_length_pixels * 8
        return total_height_pixels

    def estimate_runner_height(self, frame, results, bbox):
        """
        Estimate runner's height using head length and anthropometric proportions.
        """
        if results.pose_landmarks:
            height, width, _ = frame.shape

            # Extract required landmarks for head (top and chin)
            required_points = [0, 1]  # Top of the head, chin
            interpolated_landmarks = self.interpolate_missing_landmarks(
                results.pose_landmarks.landmark, required_points)

            # Adjust landmarks for the bounding box
            adjusted_landmarks = [(x * width + bbox[0], y * height + bbox[1]) for x, y in interpolated_landmarks]

            # Calculate height based on head length
            runner_height = self.calculate_height(adjusted_landmarks, width, height)
            return runner_height / height  # Normalize for pixel-to-meter conversion
        return None

    

    def calculate_angle(self, point_a, point_b, point_c):
        """
        Calculate the angle between three points (A, B, C).
        """
        ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
        bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])

        # Calculate the dot product and magnitudes
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)

        if magnitude_ba * magnitude_bc == 0:  # Prevent division by zero
            return 0

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        return math.degrees(angle)

    def is_upright(self, landmarks, width, height):
        """
        Determine if the runner is in an upright posture based on knee angles.
        """
        left_hip = (landmarks[23].x * width, landmarks[23].y * height)
        left_knee = (landmarks[25].x * width, landmarks[25].y * height)
        left_ankle = (landmarks[27].x * width, landmarks[27].y * height)

        right_hip = (landmarks[24].x * width, landmarks[24].y * height)
        right_knee = (landmarks[26].x * width, landmarks[26].y * height)
        right_ankle = (landmarks[28].x * width, landmarks[28].y * height)

        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Check if both knees are nearly straight (angle ~180°)
        return abs(left_knee_angle - 180) < 30 or abs(right_knee_angle - 180) < 30



    def calculate_slope_and_angle(self, ankle_left, ankle_right):
        """
        Calculate the slope and angle of the surface based on the left and right ankle positions.
        """
        delta_y = ankle_right[1] - ankle_left[1]
        delta_x = ankle_right[0] - ankle_left[0]

        if delta_x == 0:
            slope = float('inf')  # Infinite slope (vertical line)
        else:
            slope = delta_y / delta_x

        angle = math.atan(slope) * 180 / math.pi
        return slope, angle

    def refine_gradient_with_hips(self, results):
        """
        Refine the gradient calculation by checking the alignment of the hips.
        """
        # Extract ankle keypoints (left ankle: 27, right ankle: 28)
        ankle_left = (results.pose_landmarks.landmark[27].x, results.pose_landmarks.landmark[27].y)
        ankle_right = (results.pose_landmarks.landmark[28].x, results.pose_landmarks.landmark[28].y)

        # Calculate slope and angle for the ankle keypoints
        ankle_slope, ankle_angle = self.calculate_slope_and_angle(ankle_left, ankle_right)

        # Extract hip keypoints (left hip: 23, right hip: 24)
        hip_left = (results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[23].y)
        hip_right = (results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y)

        # Calculate slope and angle for the hip keypoints
        hip_slope, hip_angle = self.calculate_slope_and_angle(hip_left, hip_right)

        # Check if the hip alignment matches the ankle slope (for refinement)
        if abs(ankle_angle - hip_angle) < 10:  # 10 degrees tolerance for alignment
            return ankle_angle
        else:
            return ankle_angle
        

    def interpret_gradient(self, angle):
        """
        Interpret the gradient angle.
        """
        if -45 <= angle <= 45:
            if angle > 0:
                return "Uphill Slope"
            elif angle < 0:
                return "Downhill Slope"
            else:
                return "Flat Surface"
        else:
            return "Extreme Slope"
        

    def check_foot_contact(self, landmarks, width, height):
        """
        Check if the left and right feet are grounded. Return foot positions when they touch the ground.
        """
        # Left and right ankle positions (index 27 for left, 28 for right ankle)
        left_ankle = (landmarks[27].x * width, landmarks[27].y * height)
        right_ankle = (landmarks[28].x * width, landmarks[28].y * height)
        
        # Check if ankle y-coordinates are close to the ground level (ground contact threshold)
        ground_contact_threshold = height * 0.85  # You can adjust this value based on the frame resolution

        left_foot_contact = left_ankle[1] >= ground_contact_threshold
        right_foot_contact = right_ankle[1] >= ground_contact_threshold
        
        if left_foot_contact and not self.left_foot_grounded:
            # Store left foot contact position
            self.left_foot_position = left_ankle
            self.left_foot_grounded = True

        if right_foot_contact and not self.right_foot_grounded:
            # Store right foot contact position
            self.right_foot_position = right_ankle
            self.right_foot_grounded = True

        if self.left_foot_grounded and self.right_foot_grounded:
            # Calculate the slope when both feet are grounded
            slope, angle = self.calculate_slope_and_angle(self.left_foot_position, self.right_foot_position)
            surface_type = self.interpret_gradient(angle)
            return slope, angle, surface_type
        
        # If only one foot is grounded, return None
        return None, None, None
    
    def reset_foot_contact(self):
        """
        Reset foot contact states for the next frame.
        """
        self.left_foot_grounded = False
        self.right_foot_grounded = False
        self.left_foot_position = None
        self.right_foot_position = None




    def calculate_angle1(self, point1, point2, point3):
        """
        Calculate the angle between three points using the dot product.
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle


        return angle


    def draw_pose_with_angles(self, frame, results, bbox):
        



        if results.pose_landmarks:
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            #Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # print(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value])


           
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            # Define the points for angle calculation

            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
           




            if len(landmarks) >= 3:  # Ensure we have enough landmarks
               
                left_angle = self.calculate_angle1(left_shoulder, left_hip, left_knee)
                right_angle = self.calculate_angle1(right_shoulder, right_hip, right_knee)
                angle_difference = abs(left_angle - right_angle)
                
                # Calculate the midpoint between the left and right hips
                midpoint_x = int((left_hip[0] * width + right_hip[0] * width) / 2) + x1
                midpoint_y = int((left_hip[1] * height + right_hip[1] * height) / 2) + y1

                if angle_difference <= 5:
                        
                            # Draw the text at the midpoint
                        cv2.putText(
                            frame, "Pelvis neutral",
                            (midpoint_x, midpoint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

                elif 5 < angle_difference <= 10:
                        
                        # Draw the text at the midpoint
                        cv2.putText(
                            frame, f"Moderate difference detected. Consider checking posture and strength.",
                            (midpoint_x, midpoint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                else:
                        
                        # Draw the text at the midpoint
                        cv2.putText(
                            frame, f"Pelvic Mis-alignment: {angle_difference:.2f}",
                            (midpoint_x, midpoint_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                    


                

        return frame



    

