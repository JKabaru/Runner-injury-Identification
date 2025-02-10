from ultralytics import YOLO 

model = YOLO('models/bestly.pt')


results = model.predict('E:/Users/Public/ProjectICS/All_videos/002/5238-17_700000.avi' ,save=True)

print(results[0])
print('*************')
for box in results[0].boxes:
    print(box)