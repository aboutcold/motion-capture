import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from datetime import timedelta

# Define the timestamp for the start of the video (needs to match data)
video_start_timestamp = pd.to_datetime('2024/8/6 12:00:00')

# Calculate corresponding timestamps
start_timestamp = video_start_timestamp + timedelta(seconds=155.202152)
end_timestamp = video_start_timestamp + timedelta(seconds=179.202152)

# Read CSV file
csv_file_path = r"C:\Users\zwh10\Desktop\新建文件夹\combined_data3.csv"
df = pd.read_csv(csv_file_path, dtype={'Acc_X': float}, low_memory=False)

# Convert Timestamp column to datetime type
df['Original_Timestamp'] = pd.to_datetime(df['Original_Timestamp'])

# Extract data based on the video clip time range
extracted_data = df[(df['Original_Timestamp'] >= start_timestamp) & (df['Original_Timestamp'] <= end_timestamp)]

# Extract acceleration data
acc_x_data = extracted_data['Acc_X'].values
timestamps = extracted_data['Original_Timestamp'].values

# Calculate the number of action data points per second
fps_video = 59.94
fps_action = 120

if fps_video == 0 or fps_action == 0:
    raise ValueError("FPS values must be greater than zero")

frame_interval = int(fps_action / fps_video)  # Number of action data points corresponding to each second of video frames

if frame_interval == 0:
    raise ValueError("Frame interval calculation resulted in zero")

# Retain the action data points corresponding to each second of video frames
filtered_acc_x_data = acc_x_data[::frame_interval]

# Load video
video_path = r"C:\Users\zwh10\Desktop\clipped_video01.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Get video frame rate and total number of frames
fps_video = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {frame_width}, Frame height: {frame_height}")

# Create a window for the acceleration data plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
canvas = FigureCanvas(fig)

# Initialize plot line
line, = ax.plot([], [], 'b-', label='Acc_X')
ax.set_xlabel('Frame')
ax.set_ylabel('Acc_X')
ax.legend()

# Set output video path and parameters
output_video_path = r"C:\Users\zwh10\Desktop\combined01.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps_video, (frame_width, frame_height))

# Process video frames and update plot
for i in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break

    # Get the timestamp for the current frame
    frame_time = video_start_timestamp + timedelta(seconds=i / fps_video)

    # Calculate the corresponding action data index
    acc_index = int(i // (fps_video / fps_action))
    
    if acc_index < len(filtered_acc_x_data):
        acc_x_value = filtered_acc_x_data[acc_index]
    else:
        acc_x_value = np.nan

    # Update plot line data
    line.set_data(np.arange(i + 1), filtered_acc_x_data[:i + 1])
    ax.relim()
    ax.autoscale_view()

    # Draw the plot onto an image
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR

    # Resize the image to match the video frame
    img = cv2.resize(img, (frame_width, frame_height))

    # Overlay the image onto the video frame
    combined_frame = cv2.addWeighted(frame, 0.7, img, 0.3, 0)

    # Write the new video frame
    out.write(combined_frame)

# Release video resources
cap.release()
out.release()
plt.close(fig)

print("Annotated video has been saved.")
