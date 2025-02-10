import os

def convert_to_yolo_format(img_width, img_height, bbox):
    """
    Converts x, y, w, h to YOLO format: x_center, y_center, w, h (normalized).
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

def process_annotation_line(line):
    """
    Processes a single line of annotation: parses bbox and class name.
    """
    parts = line.strip().split('\t')
    x, y, w, h = map(int, parts[:4])  # Bounding box coordinates
    class_name = parts[4]  # "Running"
    return [x, y, w, h], class_name

def save_to_yolo_format(output_dir, image_name, bbox_list, class_id=0):
    """
    Saves bounding boxes for an image to a YOLO format .txt file.
    """
    txt_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_path, "w") as f:
        for bbox in bbox_list:
            x_center, y_center, w, h = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def merge_annotations(input_dirs, output_dir, img_width=720, img_height=404):
    """
    Merges and converts annotations from multiple input directories to YOLO format.
    """
    if not isinstance(input_dirs, list) or not input_dirs:
        print("Error: Please provide a list of input directories.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to collect all bounding boxes by file name
    combined_annotations = {}

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Error: The input directory '{input_dir}' does not exist.")
            continue
        
        print(f"Processing files in {input_dir}...")
        
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".tif.txt"):
                annotation_file = os.path.join(input_dir, file_name)
                with open(annotation_file, "r") as f:
                    lines = f.readlines()
                
                if file_name not in combined_annotations:
                    combined_annotations[file_name] = []

                for line in lines:
                    bbox, class_name = process_annotation_line(line)
                    if class_name.lower() == "running":  # Filter for the 'Running' class
                        yolo_bbox = convert_to_yolo_format(img_width, img_height, bbox)
                        combined_annotations[file_name].append(yolo_bbox)
    
    # Save all combined annotations to YOLO format
    for file_name, bbox_list in combined_annotations.items():
        if bbox_list:
            new_filename = file_name.replace('.tif.txt', '.txt')
            save_to_yolo_format(output_dir, new_filename, bbox_list)
            print(f"Saved merged YOLO labels for {new_filename}.")
        else:
            print(f"No 'Running' class found in {file_name}.")
    
    print(f"Conversion and merging completed! Labels saved in '{output_dir}'.")

# Direct execution
input_dirs = [
    'E:/Users/Public/ProjectICS/All_videos/013/gt', 
    # 'E:/Users/Public/ProjectICS/All_videos/003/gt2'
]  # List of input directories

output_dir = 'E:/Users/Public/ProjectICS/All_videos/013/labels'  # Folder to save YOLO label files

merge_annotations(input_dirs, output_dir)
