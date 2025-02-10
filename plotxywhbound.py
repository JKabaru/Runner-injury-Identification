import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Example bounding box and image
bbox = [1436.25, 361.11, 21.0, 51.0]  # [x, y, w, h]
# image = plt.imread('E:/Users/Public/ProjectICS/All_videos/001/2670-5_70111.jpg')  # Replace with a sample image from your dataset

image = plt.imread('C:/Users/joseph/Downloads/Compressed/football-players-detection.v1i.yolov11/test/images/4b770a_1_4_png.rf.5a45b3b841a06de414ceb802e34c136f.jpg')

fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw the bounding box
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Add annotations for clarity
ax.text(bbox[0], bbox[1] - 10, "Bounding Box", color='red', fontsize=10)

plt.show()
