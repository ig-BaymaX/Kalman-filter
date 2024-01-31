import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp

def kalman_filter(observed_data, initial_state, initial_estimate_error, process_variance, measurement_variance):
    num_time_steps = len(observed_data)
    filtered_states = np.zeros(num_time_steps)
    state_estimate = initial_state
    estimate_error = initial_estimate_error

    for t in range(num_time_steps):
        predicted_state = state_estimate
        predicted_estimate_error = estimate_error + process_variance
        kalman_gain = predicted_estimate_error / (predicted_estimate_error + measurement_variance)
        state_estimate = predicted_state + kalman_gain * (observed_data[t] - predicted_state)
        estimate_error = (1 - kalman_gain) * predicted_estimate_error
        filtered_states[t] = state_estimate[0]

    return filtered_states

st.title('Biomedical signal processing using Kalman Filter')

# File uploader
file = st.file_uploader("Upload a file")

# Form to enter the number of seconds
time = st.number_input(
    "Number of seconds to visualize (maximum 30 sec)", min_value=1, max_value=30, step=1)
time = int(time)

plot = st.button("Plot signal")

if file:
    # Process the file using the entered file path
    df = pd.read_csv(file, sep='\t', header=None,
                     skiprows=3000000, nrows=time*256, usecols=[29], encoding='utf_16_le')
    df1 = df[~df.apply(lambda row: row.astype(str).str.contains('AMPSAT|SHORT').any(), axis=1)]
    
    # Apply numeric conversion and drop NaN values
    df1 = df.apply(pd.to_numeric, errors='coerce')
    df1 = df1.dropna()

    # Create a time array based on a sampling rate of 256 Hz
    sampling_rate = 256
    second = np.arange(0, df1.shape[0]) / sampling_rate

    # Set initial parameters
    initial_state_estimate = 0
    initial_estimate_error = 1

    # Experiment with different values for process_variance and measurement_variance
    process_variance = 0.01  # Adjust this value
    measurement_variance = 0.1  # Adjust this value

    # Run the Kalman filter
    filtered_states = kalman_filter(observed_data=df1.values, initial_state=initial_state_estimate,
                                    initial_estimate_error=initial_estimate_error, process_variance=process_variance,
                                    measurement_variance=measurement_variance)
    filtered = hp.filtering.filter_signal(df1.iloc[:, 0], cutoff=50, sample_rate=256.0, filtertype='notch')

    # Divide the screen into 3 columns
    col1, col2, col3 = st.columns(3)
    
    # Calculate evaluation metrics
    original_signal = df1.iloc[:, 0]
    noise = original_signal - filtered_states

    # Original Signal to Noise Ratio (SNR)
    snr_original = 10 * np.log10(np.var(original_signal) / np.var(noise))
    snr_original = round(snr_original, 4)

    # Signal after applying Kalman Filter to Noise Ratio
    snr_kalman = 10 * np.log10(np.var(filtered_states) / np.var(noise))
    snr_kalman = round(snr_kalman, 4)
    

    # Original Signal to Signal after applying Kalman Filter Ratio
    snr_signal_kalman = 10 * np.log10(np.var(original_signal) / np.var(filtered_states))
    snr_signal_kalman = round(snr_signal_kalman, 4)
    

    # Display original signal in column 1
    if plot:
        with col1:
            st.subheader("Original Signal")
            plt.figure(figsize=(15, 6))
            plt.plot(second, df1.iloc[:, 0], label='Observed signal', linestyle='-', color='blue')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Signal Value')
            plt.xlabel('Time(s)')
            st.pyplot()

        # Display Kalman filtered signal in column 2
        with col2:
            st.subheader("Kalman Filtered Signal")
            plt.figure(figsize=(15, 6))
            plt.plot(second, filtered_states, label='Kalman Filtered Signal', linestyle='-', color='red')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Signal Value')
            plt.xlabel('Time(s)')
            st.pyplot()

        # Display Notch filtered signal in column 3
        with col3:
            st.subheader("Notch Filtered Signal")
            plt.figure(figsize=(15, 6))
            plt.plot(second, filtered, label='Notch Filtered Signal', linestyle='-', color='green')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Signal Value')
            plt.xlabel('Time(s)')
            st.pyplot()
            
        st.success("Plotting successful")
        st.subheader(f"Original Signal to Noise Ratio (SNR): {snr_original}")
        st.subheader(f"Signal after applying Kalman Filter to Noise Ratio: {snr_kalman}")
        st.subheader(f"Original Signal to Signal after applying Kalman Filter Ratio: {snr_signal_kalman}")
            
    
