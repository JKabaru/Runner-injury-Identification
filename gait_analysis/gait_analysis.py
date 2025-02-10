import math
import cv2

class GaitAnalysis:
    def __init__(self, pose_detector):
        self.pose_detector = pose_detector
        self.previous_left_ankle = None
        self.previous_right_ankle = None
        self.previous_left_hip = None
        self.previous_right_hip = None
        self.previous_left_knee = None
        self.previous_right_knee = None

    def track_foot_strike(self, landmarks, frame_width, frame_height):
        """
        Track when the foot makes contact with the ground using the ankle position.
        """
        left_ankle = (landmarks[27].x * frame_width, landmarks[27].y * frame_height)
        right_ankle = (landmarks[28].x * frame_width, landmarks[28].y * frame_height)

        foot_strike = None
        if self.previous_left_ankle and self.previous_right_ankle:
            # Check if the ankle position moved vertically (indicating foot strike)
            left_movement = left_ankle[1] - self.previous_left_ankle[1]
            right_movement = right_ankle[1] - self.previous_right_ankle[1]

            # Detect foot strike when vertical movement changes
            if left_movement > 5:  # Threshold to detect foot strike
                foot_strike = 'left'
            if right_movement > 5:
                foot_strike = 'right'

        # Save current ankle positions for the next frame
        self.previous_left_ankle = left_ankle
        self.previous_right_ankle = right_ankle

        return foot_strike

    def calculate_stride_length(self, landmarks, frame_width, frame_height):
        """
        Calculate stride length by measuring the horizontal distance between foot strike and the hips.
        """
        if self.previous_left_hip and self.previous_right_hip:
            # Get the hip positions
            left_hip = (landmarks[23].x * frame_width, landmarks[23].y * frame_height)
            right_hip = (landmarks[24].x * frame_width, landmarks[24].y * frame_height)

            # Calculate horizontal distance between ankle and hip for stride length
            left_stride = self.previous_left_ankle[0] - left_hip[0]
            right_stride = self.previous_right_ankle[0] - right_hip[0]

            # Determine if overstriding occurred by comparing stride length with typical range
            overstriding_left = left_stride > 0.2 * frame_width  # Adjust threshold as needed
            overstriding_right = right_stride > 0.2 * frame_width

            return left_stride, right_stride, overstriding_left, overstriding_right

        return None, None, False, False

    def calculate_hip_angle(self, landmarks, frame_width, frame_height):
        """
        Calculate the angle of the hips during running to detect overstriding.
        """
        left_hip = (landmarks[23].x * frame_width, landmarks[23].y * frame_height)
        right_hip = (landmarks[24].x * frame_width, landmarks[24].y * frame_height)
        center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Assuming the body moves horizontally, calculate angle relative to horizontal axis
        hip_angle = math.atan2(center_hip[1], center_hip[0]) * 180 / math.pi
        return hip_angle

    def calculate_knee_angle(self, landmarks, frame_width, frame_height):
        """
        Calculate knee angle (hip-knee-ankle) to identify extended or locked knee during foot strike.
        """
        left_knee = (landmarks[25].x * frame_width, landmarks[25].y * frame_height)
        left_hip = (landmarks[23].x * frame_width, landmarks[23].y * frame_height)
        left_ankle = (landmarks[27].x * frame_width, landmarks[27].y * frame_height)

        right_knee = (landmarks[26].x * frame_width, landmarks[26].y * frame_height)
        right_hip = (landmarks[24].x * frame_width, landmarks[24].y * frame_height)
        right_ankle = (landmarks[28].x * frame_width, landmarks[28].y * frame_height)

        # Calculate angle at the knee for left and right legs
        left_knee_angle = self.pose_detector.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.pose_detector.calculate_angle(right_hip, right_knee, right_ankle)

        return left_knee_angle, right_knee_angle

    def analyze_gait(self, landmarks, frame_width, frame_height):
        """
        Combine stride length, hip angle, and knee angle analysis for gait analysis.
        """
        # Step 1: Track foot strike
        foot_strike = self.track_foot_strike(landmarks, frame_width, frame_height)

        # Step 2: Calculate stride length and detect overstriding
        left_stride, right_stride, overstriding_left, overstriding_right = self.calculate_stride_length(landmarks, frame_width, frame_height)

        # Step 3: Calculate hip angle
        hip_angle = self.calculate_hip_angle(landmarks, frame_width, frame_height)

        # Step 4: Calculate knee angles
        left_knee_angle, right_knee_angle = self.calculate_knee_angle(landmarks, frame_width, frame_height)

        # Analyze results
        if overstriding_left or overstriding_right:
            overstriding_warning = "Warning: Overstriding detected"
        else:
            overstriding_warning = "Stride appears normal"

        # Return all the gait analysis information
        return {
            'foot_strike': foot_strike,
            'left_stride': left_stride,
            'right_stride': right_stride,
            'overstriding_left': overstriding_left,
            'overstriding_right': overstriding_right,
            'hip_angle': hip_angle,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'overstriding_warning': overstriding_warning
        }
