import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from utils import read_video, save_video 
from trackers import Tracker
from pose_detector import PoseDetector
from gait_analysis import GaitAnalysis
import cv2
import os
import threading
from PIL import Image, ImageTk

uploaded_file_path = ""
is_paused = False

def upload_video():
    global uploaded_file_path
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if file_path:
        uploaded_file_path = file_path
        upload_progress_bar.start(10)
        root.update()
        upload_progress_bar.stop()
        process_video_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Video Uploaded Successfully!")

def process_video():
    global uploaded_file_path
    if not uploaded_file_path:
        messagebox.showerror("Error", "No video uploaded.")
        return

    process_progress_bar.start(10)
    root.update()

    analysis_thread = threading.Thread(target=perform_analysis, args=(uploaded_file_path,))
    analysis_thread.start()

def perform_analysis(file_path):
    video_frames = read_video(file_path)
    tracker = Tracker('models/bestly.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)

    pose_detector = PoseDetector()
    gait_analyzer = GaitAnalysis(pose_detector)

    output_video_frames = []
    total_frames = len(video_frames)

    for frame_num, frame in enumerate(video_frames):
        runner_tracks = tracks['Runner'][frame_num]
        
        if isinstance(runner_tracks, list):
            runner_list = runner_tracks
        else:
            runner_list = runner_tracks.values()

        for runner_info in runner_list:
            bbox = runner_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cropped_frame = frame[y1:y2, x1:x2]
            pose_results = pose_detector.detect_pose(cropped_frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                gait_analysis = gait_analyzer.analyze_gait(landmarks, frame.shape[1], frame.shape[0])

                if gait_analysis['left_stride']:
                    cv2.putText(frame, f"Left Stride: {gait_analysis['left_stride']:.2f} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if gait_analysis['right_stride']:
                    cv2.putText(frame, f"Right Stride: {gait_analysis['right_stride']:.2f} px", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if gait_analysis['overstriding_warning']:
                    cv2.putText(frame, gait_analysis['overstriding_warning'], (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if 'hip_angle' in gait_analysis:
                    cv2.putText(frame, f"Hip Angle: {gait_analysis['hip_angle']:.2f}", (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                slope, angle, surface_type = pose_detector.check_foot_contact(landmarks, frame.shape[1], frame.shape[0])
                if surface_type:
                    surface_desc = pose_detector.interpret_gradient(angle)
                    cv2.putText(frame, f"Surface: {surface_desc}", (x1, y1 - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame = pose_detector.draw_pose(frame, pose_results, (x1, y1, x2, y2))
            frame = pose_detector.draw_pose_with_angles(frame, pose_results, (x1, y1, x2, y2))

        output_video_frames.append(frame)
        process_progress_bar['value'] = (frame_num + 1) / total_frames * 100
        root.update_idletasks()

    save_path = os.path.join("output_videos", "output_video.avi")
    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")
    save_video(output_video_frames, save_path)
    messagebox.showinfo("Info", f"Video saved to {save_path}")
    process_progress_bar.stop()
    play_button.config(state=tk.NORMAL)
    pause_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.NORMAL)
    # hidden_play_button.pack(pady=10)

def play_output_video():
    global is_paused
    cap = cv2.VideoCapture('output_videos/output_video.avi')

    while cap.isOpened():
        if is_paused:
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_canvas.create_image(0, 0, image=img, anchor=tk.NW)
        root.update_idletasks()
        root.update()

    cap.release()

def toggle_pause():
    global is_paused
    is_paused = not is_paused

def stop_video():
    global is_paused
    is_paused = True

# GUI Setup
root = tk.Tk()
root.title("AI-Powered Runner's Injury Detection")
root.geometry("800x600")
root.configure(bg="#121212")

# Create scrollable canvas
canvas = tk.Canvas(root, bg="#121212")
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#121212")

scrollable_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="n")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

center_frame = tk.Frame(scrollable_frame, bg="#121212")
center_frame.pack(pady=20)

# Title
title_label = tk.Label(center_frame, text="AI-Powered Runner's Injury Detection", font=("Arial", 20, "bold"), fg="white", bg="#121212")
title_label.pack(pady=20)

# Upload Section
upload_frame = tk.Frame(center_frame, bg="#121212")
upload_frame.pack(pady=20)

upload_button = tk.Button(upload_frame, text="Upload Video", command=upload_video, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555")
upload_button.pack(side=tk.LEFT, padx=10)

upload_progress_bar = ttk.Progressbar(upload_frame, orient="horizontal", length=300, mode="determinate")
upload_progress_bar.pack(side=tk.LEFT, padx=10)

# Processing Section
process_frame = tk.Frame(center_frame, bg="#121212")
process_frame.pack(pady=20)

process_video_button = tk.Button(process_frame, text="Process Video", command=process_video, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555", state=tk.DISABLED)
process_video_button.pack(side=tk.LEFT, padx=10)

process_progress_bar = ttk.Progressbar(process_frame, orient="horizontal", length=300, mode="determinate")
process_progress_bar.pack(side=tk.LEFT, padx=10)

# Output Video Section
output_frame = tk.Frame(center_frame, bg="#121212")
output_frame.pack(pady=20)

output_video_label = tk.Label(output_frame, text="Output Video Controls", font=("Arial", 16), fg="white", bg="#121212")
output_video_label.pack()

video_canvas = tk.Canvas(output_frame, width=640, height=360, bg="#000000")
video_canvas.pack(pady=10)

controls_frame = tk.Frame(output_frame, bg="#121212")
controls_frame.pack(pady=10)

play_button = tk.Button(controls_frame, text="Play", command=play_output_video, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555", state=tk.DISABLED)
play_button.pack(side=tk.LEFT, padx=10)

pause_button = tk.Button(controls_frame, text="Pause/Resume", command=toggle_pause, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555", state=tk.DISABLED)
pause_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(controls_frame, text="Stop", command=stop_video, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555", state=tk.DISABLED)
stop_button.pack(side=tk.LEFT, padx=10)



# hidden_play_button = tk.Button(output_frame, text="Play Output Video", command=play_output_video, font=("Arial", 12), bg="#333333", fg="white", activebackground="#555555")
# hidden_play_button.pack(pady=10)
# hidden_play_button.pack_forget()

# Footer
footer_label = tk.Label(center_frame, text="Powered by AI/ML", font=("Arial", 10), fg="white", bg="#121212")
footer_label.pack(pady=20)

root.mainloop()
