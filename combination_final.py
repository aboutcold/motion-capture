import pandas as pd
from pydub import AudioSegment
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the path for ffmpeg
os.environ["PATH"] += os.pathsep + r"C:\Users\zwh10\Downloads\ffmpeg-7.0.2-essentials_build\ffmpeg-7.0.2-essentials_build\bin"

# Read the CSV file
csv_file_path = r"C:\Users\zwh10\Desktop\Xsens DOT_D422CD009708_20240729_151636.csv"
try:
    df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_file_path, encoding='ISO-8859-1', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='cp1252', low_memory=False)

# Assume the data collection start time
start_time = pd.to_datetime('2024-08-06 12:00:00')
frequency = '8.333ms'  # The time interval for 120 times per second is 1/120 seconds

# Generate timestamps
num_rows = len(df)
timestamps = pd.date_range(start=start_time, periods=num_rows, freq=frequency)
df['Timestamp'] = timestamps

# Convert all acceleration data values to positive
df['Acc_X'] = df['Acc_X'].abs()

# Print the first few rows to confirm timestamp generation
print("CSV Data with Timestamps:")
print(df.head())

# Read the WAV file
wav_file_path = r"C:\Users\zwh10\Desktop\all_10min.wav"
audio = AudioSegment.from_wav(wav_file_path)

# Extract volume data
frame_rate = 48000
duration_seconds = audio.duration_seconds
total_samples = int(duration_seconds * frame_rate)

# Calculate volume for every 400 samples, corresponding to 120 times per second
samples_per_frame = frame_rate // 120
volume_dBFS = []

for i in range(0, total_samples, samples_per_frame):
    start_time_ms = i * 1000 // frame_rate
    end_time_ms = (i + samples_per_frame) * 1000 // frame_rate
    segment = audio[start_time_ms:end_time_ms]
    
    # Calculate volume and save
    volume_dBFS.append(segment.dBFS)

# Generate audio timestamps
timestamps_audio = pd.date_range(start=start_time, periods=len(volume_dBFS), freq=frequency)

# Create a DataFrame to store volume data
audio_features = pd.DataFrame({
    'Timestamp': timestamps_audio,
    'Volume_dBFS': volume_dBFS
})

# Print the first few rows to confirm reading and conversion
print("Audio Features Data with Timestamps:")
print(audio_features.head())

# Apply a smoothing filter (moving average)
def smooth_signal(signal, window_len=11):
    """Function to smooth the signal"""
    if window_len < 3:
        return signal
    s = np.r_[signal[window_len-1:0:-1], signal, signal[-2:-window_len-1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[window_len // 2 - 1:-window_len // 2]

audio_features['Volume_dBFS'] = smooth_signal(audio_features['Volume_dBFS'], window_len=11)

# Find multiple peaks in the volume data
volume_peaks, _ = find_peaks(audio_features['Volume_dBFS'], height=-20)
volume_peak_times = audio_features['Timestamp'].iloc[volume_peaks]

# Find multiple peaks in the acceleration data
acc_peaks, _ = find_peaks(df['Acc_X'], height=40)
acc_peak_times = df['Timestamp'].iloc[acc_peaks]

# Print peak times to check the data
print(f"Volume peak times:\n{volume_peak_times}")
print(f"Accelerometer peak times:\n{acc_peak_times}")

# Convert timestamp columns to datetime format
volume_peak_times = pd.to_datetime(volume_peak_times)
acc_peak_times = pd.to_datetime(acc_peak_times)

import pandas as pd

# Assume volume_peak_times and acc_peak_times are pandas Series of datetime type

import pandas as pd

# Assume volume_peak_times and acc_peak_times are pandas Series of datetime type

# Calculate time differences
volume_time_diffs = (volume_peak_times - volume_peak_times.iloc[0])
acc_time_diffs = (acc_peak_times - acc_peak_times.iloc[0])

# Ensure time differences are pandas Series objects
volume_time_diffs = volume_time_diffs.dt.total_seconds()
acc_time_diffs = acc_time_diffs.dt.total_seconds()

# Ensure indices start from 0
volume_time_diffs = volume_time_diffs.reset_index(drop=True)
acc_time_diffs = acc_time_diffs.reset_index(drop=True)

# Matching process
matched_volume_peaks = []
matched_acc_peaks = []
time_offsets = []

# Iterate over all accelerometer peaks
for acc_idx in range(len(acc_peak_times)):
    acc_time_diff = acc_time_diffs[acc_idx]
    
    # For each accelerometer peak, find the closest volume peak
    min_diff = float('inf')
    best_volume_idx = -1
    
    for volume_idx in range(len(volume_peak_times)):
        volume_time_diff = volume_time_diffs[volume_idx]
        diff = abs(acc_time_diff - volume_time_diff)
        
        if diff < min_diff:
            min_diff = diff
            best_volume_idx = volume_idx
    
    if best_volume_idx != -1:
        matched_volume_peaks.append(volume_peak_times.iloc[best_volume_idx])
        matched_acc_peaks.append(acc_peak_times.iloc[acc_idx])
        
        # Calculate time offset
        time_offset = acc_peak_times.iloc[acc_idx] - volume_peak_times.iloc[best_volume_idx]
        time_offsets.append(time_offset.total_seconds())

# Output results
for acc, vol, offset in zip(matched_acc_peaks, matched_volume_peaks, time_offsets):
    print(f"Accelerometer Peak: {acc}, Volume Peak: {vol}, Time Offset: {offset} seconds")

# Calculate the median time offset
median_time_offset = np.median(time_offsets) if time_offsets else None

# Adjust audio data with the best time offset
audio_features['Aligned_Timestamp'] = audio_features['Timestamp'] + pd.to_timedelta(median_time_offset, unit='s')

# Create a new DataFrame to store aligned data
aligned_data = pd.DataFrame({
    'Original_Timestamp': df['Timestamp'],
    'Acc_X': df['Acc_X'],
    'Aligned_Timestamp': audio_features['Aligned_Timestamp'],
    'Volume_dBFS': audio_features['Volume_dBFS']
})

# Save the aligned data to a CSV file
aligned_csv_path = r"C:\Users\zwh10\Desktop\新建文件夹\combined_data3.csv"
aligned_data.to_csv(aligned_csv_path, index=False, encoding='utf-8')

print(f"Aligned data has been saved to {aligned_csv_path}")

# Print matching results and time offsets
print("Matched Volume Peaks and Accelerometer Peaks:")
for volume_peak, acc_peak, offset in zip(matched_volume_peaks, matched_acc_peaks, time_offsets):
    print(f"Volume Peak: {volume_peak}, Accelerometer Peak: {acc_peak}, Time Offset: {offset} seconds")

print(f"Median Time Offset: {median_time_offset} seconds")

# Visualize the aligned dataset
plt.figure(figsize=(14, 7))

# Acceleration data
plt.plot(df['Timestamp'], df['Acc_X'], label='Accelerometer Data', color='blue')

# Volume data
plt.plot(audio_features['Aligned_Timestamp'], audio_features['Volume_dBFS'], label='Volume Data', color='red')

# Mark matched peaks
for volume_peak, acc_peak in zip(matched_volume_peaks, matched_acc_peaks):
    plt.axvline(x=volume_peak, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=acc_peak, color='blue', linestyle='--', alpha=0.5)

plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.title('Aligned Accelerometer and Volume Data')
plt.legend()
plt.grid(True)
plt.show()
