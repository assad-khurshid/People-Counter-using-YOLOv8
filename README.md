# People Counter using YOLOv8

This project implements a real-time people counting system using the YOLOv8 model. It is designed to detect and count people in a given video feed, with tracking to ensure accurate counting as individuals cross predefined lines in the frame.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The People Counter project leverages the YOLOv8 model for real-time object detection, focusing on identifying and tracking people in video feeds. The system counts people as they cross predefined lines in the video frame, with one line for upward movement and another for downward movement. A mask is applied to the video feed to focus detection on relevant areas, improving accuracy and performance.

![YOLOv8 Model Diagram](https://github.com/assad-khurshid/People-Counter-using-YOLOv8/blob/main/img.png)

## Features

- **Real-Time Detection and Counting**: Detects and counts people in real-time as they cross specified lines.
- **Bidirectional Counting**: Counts people moving in both directions (upward and downward) using separate lines.
- **Accurate Object Tracking**: The SORT algorithm ensures objects are tracked consistently across frames.
- **High Confidence Filtering**: Only objects with a confidence score above a threshold (e.g., 30%) are considered for counting.
- **Masked Region Detection**: A mask is used to focus detection on specific areas of the frame, enhancing efficiency.

## Installation

### Prerequisites

- Python 3.8+
- Anaconda (optional, but recommended)
- OpenCV
- YOLOv8 (Ultralytics)
- cvzone
- numpy
- SORT (Simple Online and Realtime Tracking)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/People-Counter-YOLOv8.git
   cd People-Counter-YOLOv8
   ```

2. **Set Up the Environment**:
   ```bash
   conda create --name people-counter python=3.8
   conda activate people-counter
   ```

3. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Weights**:
   Download YOLOv8 weights from [Ultralytics' official site](https://github.com/ultralytics/ultralytics) and place them in the `Yolo-weights` folder.

5. **Run the Application**:
   ```bash
   python people_counter.py
   ```

## Usage

1. Ensure that you have a video source (either a live camera feed or a video file) placed in the appropriate directory.
2. Customize the `limitsUp` and `limitsDown` variables in the code to define the lines where people will be counted.
3. Run the `people_counter.py` script to start counting people.
4. The system will display the video feed with detected objects, bounding boxes, and the count of people moving in each direction.

## Methodology

### Detection and Tracking

The YOLOv8 model is used for object detection, focusing specifically on people. The SORT algorithm tracks each detected person across multiple frames. When a person crosses one of the predefined lines, they are counted, with separate counters for upward and downward movement.

### Masking

A mask is applied to the video frames to ensure that only the relevant portions of the image are processed by the detection model. This improves efficiency by focusing on areas where people are expected to be present.

### Counting Logic

Two lines are defined using the `limitsUp` and `limitsDown` variables. When the center of a detected person crosses either line, they are counted, and a unique ID is assigned to prevent double counting.

## Technologies Used

- **Python 3.8**: The programming language used for the project.
- **YOLOv8 (Ultralytics)**: Primary object detection model used.
- **OpenCV**: Used for video processing and image manipulation.
- **cvzone**: Helper library to draw bounding boxes and manage object detection visuals.
- **SORT**: Algorithm used for object tracking.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

