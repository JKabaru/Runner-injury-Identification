# AI-Powered Runner's Injury Detection

## Overview
AI-Powered Runner's Injury Detection is a real-time video analysis system designed to track and analyze a runner's biomechanics to predict potential injuries. The system utilizes computer vision techniques, machine learning models, and a Tkinter-based UI for user interaction.

## Features
- **Real-time Runner Tracking**: Detects and tracks multiple runners using AI-based models.
- **Biomechanical Analysis**: Evaluates stride length, hip angles, gait asymmetry, and force distribution.
- **Surface Gradient Detection**: Identifies and classifies running surfaces for external factor analysis.
- **Tkinter UI**:
  - Upload and process videos easily.
  - View results in a separate video playback window.
  - Control video playback with play, pause, stop, normal speed, and slow-motion options.
- **Dictionary-Based Runner Tracks**: Supports data storage and retrieval for multi-runner tracking.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies from `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-runner-injury-detection.git
   cd ai-runner-injury-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. **Upload a Video**: Click on the "Upload Video" button and select a file.
2. **Process the Video**: Click on "Process Video" to analyze the runner's movement.
3. **View Results**: Click "View Results" to open a separate video window with playback controls.
4. **Control Playback**:
   - Play, pause, or stop the video.
   - Adjust playback speed (normal or slow-motion).

## Technologies Used
- **Python**: Core programming language
- **OpenCV**: Computer vision and video processing
- **YOLO & AlphaPose**: Runner detection and pose estimation
- **Tkinter**: User interface development
- **Numpy, Pandas**: Data analysis and processing

## Future Improvements
- Integration of deep learning models for better injury prediction.
- Support for live video stream analysis.
- Advanced reporting and visualization of runner performance.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-xyz`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-xyz`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or support, please contact [your-email@example.com](mailto:your-email@example.com).

