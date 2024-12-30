import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from open_ephys.analysis import Session
from scipy.signal import butter, filtfilt

###############################################################################
###############################################################################
################### THIS CODE RUNS IN 4 SEPARATE CHUNKS #######################
############# RUN EACH CHUNK INDIVIDUALLY BEFORE MOVING TO NEXT ###############
###############################################################################
###############################################################################




###############################################################################
##################### CHUNK 1: LOAD AND ORGANIZE DATA #########################
###############################################################################


# Set path and assign conditions, active channels, and filter parameters
directory = '/Users/gs075/Desktop/open_ephys_test/batch2/20241220_control_tumors'
conditions = ['injection', 'recording_site']
active_channels = [1, 2, 3, 4, 5, 6]
low_cut = 500   # Low cutoff frequency (Hz)
high_cut = 5000 # High cutoff frequency (Hz)
baseline_threshold = 4  # Initial baseline threshold (in standard deviations)
spike_detection_factor = 1.5  # Spike detection factor (Std Devs from +/- baseline)
min_spike_interval = .002  # Minimum spike interval (seconds)
display_window = .005  # 10 ms window (before and after spike)

# Get a list of all subfolders in the directory (representing blocks/recordings)
blocks = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Define the bandpass filter function
def bandpass_filter(data, low_cut, high_cut, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='bandpass', output='ba')
    return filtfilt(b, a, data)

# Process recordings function
def process_recordings(block):
    global all_data  # Ensure all_data is treated as a global variable
    
    block_path = os.path.join(directory, block)
    
    # Split the block name and extract the relevant identifiers
    temp = block.split('_')[2:]  # Skip the first two elements ('mouse_id' and 'other identifier')
    
    # Extract mouse_id (first element of temp)
    mouse_id = temp[0]
    mouse_id_dict = {'mouse_id': [mouse_id] * len(active_channels)}

    # Extract condition values (remaining elements in temp)
    condition_values = temp[1:]
    conditions_columns = {condition: [condition_values[i]] * len(active_channels) for i, condition in enumerate(conditions)}

    # Access the session and data
    session = Session(block_path)
    recordnode = session.recordnodes[0]
    recording = recordnode.recordings[0]
    fs = recording.info['continuous'][0]['sample_rate']
    fs_columns = {'fs': [fs] * len(active_channels)}

    # Access the continuous data
    data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=100000000)  # This should return a numpy array or similar structure
    data = data.T  # Transpose the data to get channels as rows

    # Extract the active channels
    unfiltered_data = {'unfiltered_data': [data[channel-1] for channel in active_channels]}  # Convert to 0-based index

    # Filter the data
    filtered_data = []
    for i, data in enumerate(unfiltered_data['unfiltered_data']):
        filtered_data.append(bandpass_filter(data, low_cut, high_cut, fs))
    filtered_data = {'filtered_data': filtered_data}

    # Combine everything into a single dictionary
    final = {**mouse_id_dict, **conditions_columns, **fs_columns, **unfiltered_data, **filtered_data}
    
    # Convert to a DataFrame
    final_df = pd.DataFrame(final)

    # Concatenate the current final DataFrame with the all_data DataFrame
    global all_data
    all_data = pd.concat([all_data, final_df], axis=0, ignore_index=True)

# Process the recordings for each block
print('loading data')
all_data = pd.DataFrame()
for block in blocks:
    process_recordings(block)

# Baseline detection function
def detect_baseline(channel_data, fs, baseline_threshold=baseline_threshold):
    """Detects baseline and calculates thresholds."""
    baseline_mean = np.mean(channel_data)
    baseline_std = np.std(channel_data)

    # Define baseline thresholds based on the baseline
    baseline_upper = baseline_mean + baseline_threshold * baseline_std
    baseline_lower = baseline_mean - baseline_threshold * baseline_std

    return baseline_upper, baseline_lower, baseline_mean, baseline_std

# Spike detection function
def detect_spikes(channel_data, fs, baseline_upper, baseline_lower, spike_detection_factor, min_spike_interval):
    spike_threshold_upper = baseline_upper + spike_detection_factor * np.std(channel_data)
    spike_threshold_lower = baseline_lower - spike_detection_factor * np.std(channel_data)
    
    spikes = []
    last_spike_time = -min_spike_interval  # Ensure we start with a time before the first spike
    for i in range(1, len(channel_data)):
        if (channel_data[i] > spike_threshold_upper or channel_data[i] < spike_threshold_lower):
            time_in_seconds = i / fs  # Convert index to time in seconds
            # Ensure there's a minimum interval between detected spikes
            if time_in_seconds - last_spike_time >= min_spike_interval:
                spikes.append(i)  # Store the index of the spike
                last_spike_time = time_in_seconds
    return spikes





data = all_data
# Function to assign 'Channel' based on the active channels
def assign_channel_number(row):
    # Find the channel number for this row (1-based index for active_channels)
    return f"Channel {active_channels[row.name % len(active_channels)]}"

# Apply the function to create the new 'channel_number' column
data['channel_number'] = data.groupby(['mouse_id'] + conditions).apply(lambda x: [f"Channel {active_channels[i % len(active_channels)]}" for i in range(len(x))]).explode().reset_index(drop=True)

# Show the first few rows of the dataframe to check the results

def apply_spike_detection(data, spike_detection_factor=spike_detection_factor, min_spike_interval=min_spike_interval):
    def detect_and_store_spikes(row):
        # Extract channel data for this row
        channel_data = row['filtered_data']
        fs = row['fs']  # Get the sample rate for this row

        # Detect baseline thresholds for spike detection
        baseline_upper, baseline_lower, _, _ = detect_baseline(channel_data, fs, baseline_threshold)

        # Detect spikes using the thresholds
        spikes = detect_spikes(channel_data, fs, baseline_upper, baseline_lower, spike_detection_factor, min_spike_interval)

        return spikes

    # Apply the function across all rows and store detected spikes
    data['spikes'] = data.apply(detect_and_store_spikes, axis=1)
    return data


data=apply_spike_detection(data)






























###############################################################################
######## CHUNK 2: PLOT AND VERIFY SIGNAL + BASELINE AND THRESHOLDING ##########
###############################################################################


# Function to plot the filtered signal with baseline and spike detection thresholds
def plot_signal(channel_index):
    global current_channel_index, baseline_threshold, spike_detection_factor, min_spike_interval

    # Get the data for the selected channel from the 'data' DataFrame
    channel_data = data['filtered_data'].iloc[channel_index]  # Channel signal data
    fs = data['fs'].iloc[channel_index]  # Sampling frequency in Hz
    channel_number = data['channel_number'].iloc[channel_index]
    # Detect baseline thresholds (upper and lower) for spike detection
    baseline_upper, baseline_lower, baseline_mean, baseline_std = detect_baseline(channel_data, fs, baseline_threshold)
    
    # Calculate spike detection thresholds
    spike_detection_upper = baseline_upper + spike_detection_factor * baseline_std
    spike_detection_lower = baseline_lower - spike_detection_factor * baseline_std

    # Extract the spike timestamps for the current channel

    # Clear the axes without clearing the entire figure
    plt.gca().cla()

    # Convert the sample index to time in milliseconds (time = sample_index / fs * 1000)
    time_ms = np.arange(len(channel_data)) / fs * 1000  # Time in milliseconds

    # Plot the full filtered signal
    plt.plot(time_ms, channel_data, label='Filtered Signal', color='b', alpha=0.7)

    # Plot baseline upper and lower thresholds
    plt.axhline(y=baseline_upper, color='r', linestyle='--', label='Baseline Upper')
    plt.axhline(y=baseline_lower, color='g', linestyle='--', label='Baseline Lower')

    # Plot spike detection upper and lower thresholds
    plt.axhline(y=spike_detection_upper, color='b', linestyle=':', label='Spike Detection Upper Threshold')
    plt.axhline(y=spike_detection_lower, color='m', linestyle=':', label='Spike Detection Lower Threshold')

    # Plot detected spikes as vertical lines (at spike timestamps) **outside** the loop
    
    # Add labels and title
    mouse_id = data['mouse_id'].iloc[channel_index]
    condition_values = {condition: data[condition].iloc[channel_index] for condition in conditions}
    identifier_info = f"Mouse ID: {mouse_id}, " + ", ".join([f"{k}: {v}" for k, v in condition_values.items()])
    plt.title(f"{channel_number} - {identifier_info}", fontsize=10)

    plt.xlabel("Time (ms)")  # X-axis in milliseconds
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)

    # Set y-limits dynamically based on signal amplitude to ensure full visibility
    plt.ylim([min(channel_data) - 10, max(channel_data) + 10])  # Adjusting y-axis for better visibility

    # Show the plot
    plt.legend(loc='upper right', fontsize=8)
    plt.draw()


# Handle keypress events for channel navigation
def on_key(event):
    global current_channel_index

    # Left arrow (previous channel)
    if event.key == 'left' and current_channel_index > 0:
        current_channel_index -= 1

    # Right arrow (next channel)
    elif event.key == 'right' and current_channel_index < len(data) - 1:
        current_channel_index += 1

    # Update the plot with the new channel
    plot_signal(current_channel_index)

# Initialize the plot with the first channel
current_channel_index = 0
plot_signal(current_channel_index)

# Attach keypress event to navigate between channels
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

# Show the plot
plt.show()






















###############################################################################
########## CHUNK 3: PLOT AND VERIFY THRESHOLDED ACTION POTENTIALS #############
###############################################################################
current_spike_index = 0


def plot_spike_waveform(spike_index):
    global current_channel_index, display_window_ms, baseline_threshold, spike_detection_factor
    
    # Get the data for the selected channel
    channel_data = data['filtered_data'].iloc[current_channel_index]
    fs = data['fs'].iloc[0]  # Sampling frequency in Hz
    
    # Extract the spike timestamps for the current channel (in samples)
    spike_times = data['spikes'].iloc[current_channel_index]
    
    # Ensure the spike_index is valid
    if spike_index < 0 or spike_index >= len(spike_times):
        print(f"Invalid spike index: {spike_index}")
        return
    
    # Get the spike timestamp (in samples) for the selected spike index
    spike_sample_index = spike_times[spike_index]  # Timestamp is already in samples
    
    # Create a time window (display_window_ms before and after the spike)
    window_size = int(display_window * fs )
    start_sample = max(spike_sample_index - window_size, 0)  # Start of the window (before the spike)
    end_sample = min(spike_sample_index + window_size, len(channel_data))  # End of the window (after the spike)
    
    
    # Extract the signal segment (spike window) around the spike
    spike_window = channel_data[start_sample:end_sample]
    
    # Check if the spike window is empty
    if len(spike_window) == 0:
        print("Error: Empty spike window!")
        return
    
    # Time in milliseconds for the current window around the spike
    time_window_ms = np.arange(start_sample, end_sample) / fs * 1000  # Convert sample indices to time in ms
    
    # Clear the axes without clearing the entire figure
    plt.gca().cla()
    
    # Plot the spike waveform for the current spike (zoomed-in around the spike)
    plt.plot(time_window_ms, spike_window, color='k', alpha=0.7)
    
    # Plot baseline thresholds (upper and lower) and spike detection thresholds
    baseline_upper, baseline_lower, baseline_mean, baseline_std = detect_baseline(channel_data, fs, baseline_threshold)
    spike_detection_upper = baseline_upper + spike_detection_factor * baseline_std
    spike_detection_lower = baseline_lower - spike_detection_factor * baseline_std
    
    # Plot baseline thresholds and spike detection thresholds
    plt.axhline(y=baseline_upper, color='r', linestyle='--', label='Baseline Upper')
    plt.axhline(y=baseline_lower, color='g', linestyle='--', label='Baseline Lower')
    plt.axhline(y=spike_detection_upper, color='b', linestyle=':', label='Spike Detection Upper Threshold')
    plt.axhline(y=spike_detection_lower, color='m', linestyle=':', label='Spike Detection Lower Threshold')
    
    # Add labels and title
    mouse_id = data['mouse_id'].iloc[current_channel_index]
    chn = data['channel_number'].iloc[current_channel_index]
    condition_values = {condition: data[condition].iloc[current_channel_index] for condition in conditions}
    identifier_info = f"Mouse ID: {mouse_id}, "  + ", ".join([f"{k}: {v}" for k, v in condition_values.items()])
    plt.title(f"Spike Waveform - {identifier_info}, {chn}, Spike {spike_index + 1}", fontsize=10)

    
    plt.xlabel("Time (ms)")  # X-axis in milliseconds
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    
    # Show the plot
    plt.legend(loc='upper right', fontsize=8)
    plt.draw()
    plt.pause(0.001)  # Ensure the plot is updated interactively


def on_key(event):
    global current_channel_index, current_spike_index

    # Get the number of spikes for the current channel
    spike_times = data['spikes'].iloc[current_channel_index]
    num_spikes = len(spike_times)  # Number of spikes in the current channel
    
    # Handle keypress events:
    
    # Left arrow (previous spike)
    if event.key == 'left':
        if current_spike_index > 0:
            current_spike_index -= 1
        else:
            # Move to the previous channel, reset to the last spike
            if current_channel_index > 0:
                current_channel_index -= 1
                current_spike_index = len(data['spikes'].iloc[current_channel_index]) - 1
    
    # Right arrow (next spike)
    elif event.key == 'right':
        if current_spike_index < num_spikes - 1:
            current_spike_index += 1
        else:
            # Move to the next channel, reset to the first spike
            if current_channel_index < len(data['spikes']) - 1:
                current_channel_index += 1
                current_spike_index = 0

    # Ensure the spike index is valid before plotting
    plot_spike_waveform(current_spike_index)


# Initialize the plot with the first spike
plot_spike_waveform(current_spike_index)

# Attach keypress event to navigate between spikes
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

# Show the plot and enter the event loop
plt.show()


















###############################################################################
########## CHUNK 4: ORGANIZE AND EXPORT SUMMARY DATA FOR STATISTICS ###########
###############################################################################


individual_channel_data = data.drop(columns=['unfiltered_data', 'filtered_data', 'fs'])
individual_channel_data['spike_count'] = data['spikes'].apply(len)

individual_data_averaged = individual_channel_data.groupby(['mouse_id'] + conditions)['spike_count'].mean().reset_index()
grouped_data =individual_channel_data.groupby(conditions)['spike_count'].mean().reset_index()
grouped_data['sample_size'] = individual_channel_data.groupby(conditions)['mouse_id'].nunique().reset_index()['mouse_id']

# individual_data_averaged.to_csv(os.path.join(directory, 'individual_data_averaged.csv'))
# individual_channel_data.to_csv(os.path.join(directory, 'individual_channel_data.csv'))
# grouped_data.to_csv(os.path.join(directory, 'grouped_data.csv'))




