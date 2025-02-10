import matplotlib.pyplot as plt
import matplotlib.patches as patches

# YOLO bounding box format: class_id x_center y_center width height
# yolo_bbox = [0, 0.864584, 0.560938, 0.114583, 0.098438]  # Normalized values
yolo_bbox = [0, 0.384722, 0.557292, 0.233333, 0.510417]  # Normalized values
# bbox = [1436.25, 361.11, 21.0, 51.0]  # [x, y, w, h]
# 0 0.864584 0.560938 0.114583 0.098438   train

# 0 0.430556 0.810644 0.061111 0.155941  train1

img_width, img_height = 720, 576 # Image dimensions
image_path = 'E:/Users/Public/ProjectICS/All_videos/003/5238-17_700641.jpg'
# image_path = 'C:/Users/joseph/Downloads/Compressed/football-players-detection.v1i.yolov11/test/images/4b770a_1_4_png.rf.5a45b3b841a06de414ceb802e34c136f.jpg'

# Extract values from YOLO format
_, x_center, y_center, w, h = yolo_bbox

# Convert YOLO format to pixel format
x_top_left = (x_center - w / 2) * img_width
y_top_left = (y_center - h / 2) * img_height
bbox_width = w * img_width
bbox_height = h * img_height

# Load the image
image = plt.imread(image_path)

# Visualize the image and bounding box
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw the bounding box
rect = patches.Rectangle((x_top_left, y_top_left), bbox_width, bbox_height, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Add annotations
ax.text(x_top_left, y_top_left - 10, "YOLO Bounding Box", color='red', fontsize=10)

plt.show()
