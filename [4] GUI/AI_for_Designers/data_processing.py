from __future__ import annotations

from typing import Any

import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

from collections.abc import Collection, Iterable


class Preprocessing:
    def __init__(self, action_ID: str) -> None:
        self.action_ID = action_ID
        self.output_file = fr'Preprocessed-data/{action_ID}/features_{action_ID}.txt'
        if not os.path.exists(f'Preprocessed-data/{action_ID}'):
            os.mkdir(f'Preprocessed-data/{action_ID}')

    @staticmethod
    def time(data: Collection, samples_window: float) -> list[list[float]]:
        """Function to extract feature from the time-domain data.

        Args:
            data (Collection): array of multiple sensors. Rows are data, columns are different sensors.
            samples_window (float): the amount of samples per window.

        Returns:
            tuple[float, float, float, float, float]: tuple of features for this specific windowed sample.
        """

        # Making sure that the data collection is a numpy array     
        data = np.array(data)

        # List to save the extracted features in. It has as many items (lists) as sensors present
        features: list[list[float]] = [[] for _ in range(data.shape[1])]
        for i in range(data.shape[1]):
            # Extract the features: minimum, maximum, average, standard deviation and area under the curve
            features[i].append(min(data[:, i]))
            features[i].append(max(data[:, i]))
            features[i].append(float(np.mean(data[:, i])))
            features[i].append(float(np.std(data[:, i])))
            features[i].append(sum(data[:, i]) / samples_window)

        return features

    @staticmethod
    def fourier(data: Collection, sampling_frequency: float, epsilon: float = 0.1, zero_padding: int = 0) -> \
            tuple[Any, list[tuple[int, float, float, float]]]:
        """In the function: zero padding, fft time data columns, spectral analysis.

        Args:
            data (Collection): array of multiple sensors. Rows are data, columns are different sensors.
            sampling_frequency (float): floating point number with the sample frequency used to measure the data.
            epsilon (float): relative boundary for what values that are higher than the epsilon * maximum value is used
            to add to the total power.
            zero_padding: total amount of zeros are added to the data collection. This will increase the amount of
            frequencies in the fourier transform.

        Returns:
            tuple[np.array[float]]: tuple of features for this specific windowed sample.
        """

        # Making sure that the data collection is a numpy array
        data = np.array(data)
        # Adding the zero's if this should be done 
        if zero_padding > 0:
            data = np.append(data, np.zeros([zero_padding, 3]), axis=0)

        # Calculating the ffts
        ffts = np.fft.fft2(data, axes=(-2, -2, -2))

        # Extracting the features
        peaks = Preprocessing.peak_value_frequency(ffts[:np.shape(data)[0] // 2], sampling_frequency)
        pwrs = Preprocessing.cntrd_pwr(ffts[:np.shape(data)[0] // 2], sampling_frequency, epsilon)

        return ffts, [peak + pwr for peak, pwr in zip(peaks, pwrs)]

    @staticmethod
    def cntrd_pwr(dataset: Collection, sampling_frequency: float, epsilon: float = 0.1) -> list[tuple[float, float]]:
        """Finds the maximum power of 3 sensors and the frequency for which the energy at the left is the same as the
        energy on the right takes a threshold of epsilon*max_value.

        Args:
            dataset (Collection): frequency domain dataset.
            epsilon (float): parameter to determine what samples play a part in the centroid calculations. range 0-1.
            sampling_frequency (float): sampling frequency.

        Returns:
            list[tuple[int, int, int], tuple[float, float, float]]: list of a tuple of the centroid frequency and 
            the maximum power of all sensors.
        """

        # BIG SIDENOTE. We talk about power. THIS IS WRONG. It should be called the energy of the signal. 

        # Making sure that the dataset is a numpy array
        data = np.array(dataset)

        sensor_range = data.shape[1]
        total_power = [0.] * sensor_range  # 3 sensors total power
        centroid = [0.] * sensor_range  # 3 sensors centroid, not returned
        index = [0] * sensor_range  # index of the centroid
        # compute for all three sensors
        for i in range(sensor_range):
            maxm = max(data[1:, i])
            length = len(data[:, i])
            # Sum power and sum all values above the threshold
            for j in range(1, length):
                total_power[i] += abs(data[j][i])
                if abs(data[j][i]) > epsilon * maxm:
                    centroid[i] += abs(data[j][i])
            goal = centroid[i] / 2
            centroid[i] = 0.
            # reset j, go through the dataset again and stop when you surpass centroid/2
            j = 1
            while centroid[i] < goal:
                if abs(data[j][i]) > epsilon * maxm:
                    centroid[i] += abs(data[j][i])
                j += 1
                index[i] += 1

        return [(e * sampling_frequency / (2 * len(data)), pwr) for e, pwr in zip(index, total_power)]

    @staticmethod
    def peak_value_frequency(dataset: Collection, sampling_frequency: float) -> list[tuple[int, float]]:
        """Find the frequency that is the most present in a PSD. Does not include the DC component.

        Args:
            dataset (Collection): frequency domain dataset.
            sampling_frequency (float): the sampling frequency used to measure the data.

        Returns:
            list[tuple[int, float]]: list with a tuple containing the index and the corresponding frequency.
        """

        # List to save the peaks and their frequencies in
        peaks: list[tuple[int, float]] = []
        # Casting the dataset collection into a numpy array
        data = np.array(dataset)

        # Finding the sample in the middle of the collection (second half is a mirrored copy of the first half)
        mid = data.shape[0] // 2
        # Finding the maximum value and its location and its frequency for each column of the array
        for i in range(data.shape[1]):
            max_value = max(abs(data[1:mid, i]))
            index = np.where(abs(data[1:mid, i]) == max_value)[0][0] + 1
            peaks.append((index, index * sampling_frequency / (2 * data.shape[0])))
        return peaks

    @staticmethod
    def get_sampling_frequency(input_file: str, start_offset: float = 0, stop_offset: float = 0, size: float = 2,
                               skip_n_lines_at_start: int = 0) -> tuple[float, float, float]:
        """Retrieving the sampling frequency from the timestamps from the input file.

        Args:
            input_file: (str): file path to the data.
            start_offset (float, optional): skip the first r seconds of the data. Defaults to 0.
            stop_offset (float, optional): skip the last r seconds of the data. Defaults to 0.
            size(float, optional): size of the window in seconds. Defaults to 2.
            skip_n_lines_at_start (int, optional): skip the first n lines at the start of the data file. Sometimes the
            first few lines contain no data. These lines should be skipped. Defaults to 0.

        Returns:
            tuple[float, float, float]: returns the sampling frequency, the last timestamp and the window size.
        """

        with open(input_file) as f:
            # Check the file extension
            file_extension = input_file.split('.')[-1]
            if file_extension == 'txt':
                # Skip the header lines
                for _ in range(skip_n_lines_at_start):
                    f.readline()
            elif file_extension == 'csv':
                # Skip the header lines, and the first line
                for _ in range(2):
                    f.readline()
            # If the file extension is not supported
            else:
                raise NotImplementedError(f"Filetype {input_file.split('.')[-1]} is not implemented")

            # Find the sample frequency by dividing 1 by the difference in time of two samples
            t0 = float(f.readline().strip().split(',')[0])
            t1 = float(f.readline().strip().split(',')[0])
            sampling_frequency = round(1 / (t1 - t0), 2)

            # Finding the last timestamp of the data
            last_point = 0.0
            for line in f:
                last_point = float(line.strip().split(',')[0])

            if size + start_offset + stop_offset > last_point - t0:
                size = last_point - t0 - start_offset - stop_offset

            return sampling_frequency, last_point, size

    def windowing(self, input_file: str | list[str], video_file: str = '', label: str = '',
                  start_offset: float = 0, stop_offset: float = 0,
                  size: float = 2, offset: float = 0.2, start: int = 1, stop: int = 3, video_offset: float = 0,
                  epsilon: float = 0.1, skip_n_lines_at_start: int = 0, do_plot: bool = False, do_scale=False) -> None:
        """Function for making windows of a certain size, with an offset. A window is made and the features of the window
        are extracted. The window is slided the offset amount of seconds to the right and new window is made and a
        its features are extracted. This is done until the end of the file is reached. The extracted features are saved
        in a file with the name of the output_file attribute.

        Args:
            input_file (str, list[string]): the relative path of the file with the data, seen from the main file. If
            accelerometer data and gyroscope data both want to be used, a list with both paths can be used. Preferably
            enter the file with the accelerometer first.
            video_file (str, optional): The relative path of the file with the corresponding video the captures that
            process of capturing the data, seen from the main file. The path is only printed to an output file.
            Defaults to ''.
            label (str): the name of the label of the activity. Defaults to ''.
            size (float, optional): size of the window in seconds. Defaults to 2.
            start_offset (float, optional): skip the first r seconds of the data. Defaults to 0.
            stop_offset (float, optional): skip the last r seconds of the data. Defaults to 0.
            offset (float, optional): size of the offset in seconds. Defaults to 0.2.
            start (int, optional): start column from the data file. Defaults to 1.
            stop (int, optional): last column (including) of the data file. Defaults to 3.
            video_offset (float, optional): time in seconds that the video starts before the start of the data. Defaults to 0.
            epsilon (float, optional): variable for the fourier function. Defaults to 0.1.
            skip_n_lines_at_start (int, optional): skip the first n lines at the start of the data file. Sometimes the first
            few lines contain no data. These lines should be skipped. Defaults to 0.
            do_plot (bool, optional): set to true when a plot of every window is wanted. Defaults to False.
            do_scale (bool, optional): set to true when scaling is wanted. All the features in the feature file will be
            scaled. ATTENTION! Only set to true, when the last data is added. Otherwise, the previous files will get
            scaled multiple times. Defaults to False.

        Raises:
            NotImplementedError: this error is raised if the file extension is not supported.
            FileNotFoundError: this error is raised if the file-parameter cannot be found.
            ValueError: this error is raised when a value in the file cannot be converted to a float.
        """

        if start_offset < 0:
            raise ValueError(f'start_offset should be greater or equal to zero, but is {start_offset}')
        if stop_offset < 0:
            raise ValueError(f'stop_offset should be greater or equal to zero, but is {stop_offset}')
        if size <= 0:
            raise ValueError(f'Frame size should be greater than zero, but is {size}')
        if offset <= 0:
            raise ValueError(f'Frame offset should be greater than zero, but is {offset}')
        if size < offset:
            warnings.warn('It is advised to keep the frame size larger than the frame offset!')
        if not os.path.exists(video_file):
            raise FileExistsError(f'File at path {video_file} does not exist! Check the path')

        input_file_b = ''  # empty string to check if two files are used or not
        # Assign input_file_a and _b
        if isinstance(input_file, str):
            input_file_a = input_file
        elif len(input_file) == 1:
            input_file_a = input_file[0]
        elif len(input_file) == 2:
            input_file_a = input_file[0]
            input_file_b = input_file[1]

        print_timestamps = []

        # ID of the datapoint, necessary for active learning
        current_ID = 0
        last_index = 0

        # Counter for keeping the timestamps comparable with the timestamps list.
        # This list is used when writing to the file to know when a window starts.
        timestamp_list: list[float] = []

        # Check if the file that we want to extract the data from has already been used in this action_ID
        already_processed = False
        try:
            with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt') as f:
                for line in f:
                    if line.strip().split(',')[0] == input_file_a:
                        already_processed = True
                        break
        except FileNotFoundError:
            pass

        # This section lets you input y/n if you want to write the features to the file.
        # Prevent adding the same data twice
        while True:
            # Check if the file was already processed, if it is, ask if the file should be processed again.
            if already_processed:
                print('The file is already processed at least once.\n')
            # write_to_file = input(f"Do you want to save the extracted features '{label}' for '{self.action_ID}'? y/n\n")
            write_to_file = 'y'
            # Check if the input is valid
            if write_to_file == 'y':
                with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt', 'a') as f:
                    f.write(f"{input_file_a}")
                break
            elif write_to_file == 'n':
                break
            else:
                print("Input not valid! Try again")
           
            

        # Check if the file that we will write to exist and if it does, if it contains a header already.
        # Add header if it is not yet in the file.
        # We also check what the last number is in the file. This is used for the datapoint ID in the write-file
        try:
            with open(self.output_file, 'r') as check_file:
                if check_file.readline().strip() == '':
                    make_header = True
                else:
                    make_header = False
                    for line in check_file:
                        last_index = int(line.strip().split(',')[0]) + 1
        except FileNotFoundError:
            make_header = True

        write_to_file = 'y'
        # If we want to write and there is no header yet, build the header
        if write_to_file == 'y' and make_header:
            with open(self.output_file, 'a') as g:
                # Build list of possible labels
                sensor_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
                feature_names = ['min', 'max', 'avg', 'std', 'AUC', 'pk', 'cn', 'pw']

                full_header_features = []
                for i in range(len(sensor_names)):
                    full_header_features.append('')
                    for j in feature_names:
                        full_header_features[i] += f',{sensor_names[i]}_{j}'

                # Build the header
                specified_header = 'ID,label,time'
                for i in range(stop):
                    specified_header += full_header_features[i]
                if input_file_b != '':
                    for i in range(3, 3 + stop):
                        specified_header += full_header_features[i]

                # Write the header to file
                g.write(specified_header + '\n')

        # get sampling frequency and the last point
        sampling_frequency, last_point, size_a = self.get_sampling_frequency(input_file_a, start_offset, stop_offset,
                                                                             size)
        # Amount of samples per window
        samples_window = int(size_a * sampling_frequency) + 1  # Ceil

        # Do the same for gyroscope and call it 'b'
        if input_file_b != '':
            sampling_frequency_b, last_point_b, size_b = self.get_sampling_frequency(input_file_b, start_offset,
                                                                                     stop_offset, size)

            prev_window_b: list[list[float]] = []
            current_window_b: list[list[float]] = []

            samples_window_b = int(size_b * sampling_frequency_b) + 1

        if start_offset > last_point - stop_offset:
            raise ValueError(f'start_offset or stop_offset too large!'
                             f'The start_offset should be smaller than the last datapoint minus the stop_offset'
                             f'({last_point - stop_offset}s)')
        if input_file_b != '':
            if start_offset > last_point_b - stop_offset:
                raise ValueError(f'start_offset or stop_offset too large!'
                                 f'The start_offset should be smaller than the last datapoint minus the stop_offset'
                                 f'({last_point_b - stop_offset}s)')

        try:
            # Variable for the previous window and the current window
            prev_window: list[list[float]] = []
            current_window: list[list[float]] = []

            # When the end of a datafile is reached, this value is set to False and the loop is exited
            not_finished = True

            # Opening the data file again and skipping the header lines.
            k = 0
            with open(input_file_a) as f:
                for _ in range(skip_n_lines_at_start):
                    f.readline()
                # Open file_b if it exists:
                if input_file_b != '':
                    f_b = open(input_file_b)
                    for _ in range(skip_n_lines_at_start):
                        f_b.readline()

                line = f.readline().strip().split(',')
                while float(line[0]) < start_offset:
                    line = f.readline().strip().split(',')
                if input_file_b != '':
                    line_b = f_b.readline().strip().split(',')
                    while float(line_b[0]) < start_offset:
                        line_b = f_b.readline().strip().split(',')
                # Opening the output file; the extracted features will be put in the file
                with open(self.output_file, 'a') as g:
                    # While the end of the file is not yet reached
                    while not_finished:
                        if len(current_window) == 0:
                            startpoint = start_offset
                            timestamp_list.append(startpoint)

                            current_window.append([])
                            for i in range(start, stop + 1):
                                current_window[-1].append(float(line[i]))

                            while float(line[0]) < startpoint + size_a:
                                # Store a list of the sensordata of the line that is read
                                line = f.readline().strip().split(',')
                                # Initialise the current window, make sure the added value is an empty list
                                current_window.append([])
                                for i in range(start, stop + 1):
                                    current_window[-1].append(float(line[i]))

                            startpoint += offset
                        else:
                            # If this is the first time that previous window is called, initialise it as a copy of
                            # current window. Use .copy() to prevent aliasing
                            if len(prev_window) == 0:
                                for i in range(len(current_window)):
                                    prev_window.append(current_window[i].copy())
                            # Else we make it a direct copy of the current window as well
                            else:
                                if len(current_window) <= len(prev_window):
                                    for i in range(len(current_window)):
                                        prev_window[i] = current_window[i].copy()
                                    while len(current_window) < len(prev_window):
                                        prev_window.pop(-1)
                                else:
                                    for i in range(len(prev_window)):
                                        prev_window[i] = current_window[i].copy()
                                    while len(current_window) > len(prev_window):
                                        i += 1
                                        prev_window.append(current_window[i].copy())

                            # Overwrite the current window values with its previous values the current window is slided
                            # the samples_offset amount of samples into the future. Thus, the samples [samples_offset:]
                            # of prev_window are the first samples for the current window. The rest of the samples are
                            # new and read from the data file
                            i = 0
                            for lst in prev_window:
                                if lst[0] >= startpoint:
                                    current_window[i] = lst.copy()
                                    i += 1

                            # Read new lines from the file and add these to the end of the current file
                            try:
                                while float(line[0]) < startpoint + size_a:
                                    # The last line of the file is an empty string. When detected we exit the while loop
                                    if line[0] == '':
                                        not_finished = False
                                        break
                                    # Check if the line that you read will not be able to become a full window because
                                    # of the stop offset
                                    elif float(line[0]) > last_point - stop_offset + offset:
                                        not_finished = False
                                        break
                                    # Build new part of the frame
                                    if i > len(current_window) - 1:
                                        current_window.append([])
                                        for j in range(start, stop + 1):
                                            current_window[-1].append(0.0)
                                    # Read samples_offset amount of samples and add these to the current window
                                    for j in range(start, stop + 1):
                                        current_window[i][j - start] = float(line[j])
                                    i += 1
                                    line = f.readline().strip().split(',')
                                    if line[0] == '':
                                        not_finished = False
                                        break
                                # Make sure all windows have the same length
                                while i < len(current_window) - 1:
                                    current_window.pop(-1)

                            except EOFError:
                                not_finished = False

                        # For gyro data, code is the same as above except it does not contain have to build the timestamp
                        # list
                        if input_file_b != '':
                            # for readability check comments above, this is repeated code
                            if len(current_window) == 0:
                                current_window_b.append([])
                                for i in range(start, stop + 1):
                                    current_window_b[-1].append(float(line_b[i]))

                                while float(line_b[0]) < startpoint + size_b:
                                    line_b = f.readline().strip().split(',')
                                    current_window_b.append([])
                                    for i in range(start, stop + 1):
                                        current_window_b[-1].append(float(line_b[i]))

                            else:
                                if len(prev_window_b) == 0:
                                    for i in range(len(current_window_b)):
                                        prev_window_b.append(current_window_b[i].copy())
                                # Else we make it a direct copy of the current window as well
                                else:
                                    if len(current_window_b) <= len(prev_window_b):
                                        for i in range(len(current_window_b)):
                                            prev_window_b[i] = current_window_b[i].copy()
                                        while len(current_window_b) < len(prev_window_b):
                                            prev_window_b.pop(-1)
                                    else:
                                        for i in range(len(prev_window_b)):
                                            prev_window_b[i] = current_window_b[i].copy()
                                        while len(current_window_b) > len(prev_window_b):
                                            i += 1
                                            prev_window_b.append(current_window_b[i].copy())

                                # Overwrite the current window values with it's previous values the current window is
                                # slided the samples_offset amount of samples into the future. Thus, the samples
                                # [samples_offset:] of prev_window are the first samples for the current window. The
                                # rest of the samples are new and read from the data file
                                i = 0
                                for lst in prev_window_b:
                                    if lst[0] >= startpoint:
                                        current_window_b[i] = lst.copy()
                                        i += 1

                                # Read new lines from the file and add these to the end of the current file
                                try:
                                    while float(line_b[0]) < startpoint + size_b:
                                        # The last line of the file is an empty string. When detected we exit the while
                                        # loop
                                        if line_b[0] == '':
                                            not_finished = False
                                            # print('It stopped at gyro data', timestamp_list)
                                            break
                                        elif float(line_b[0]) > last_point - stop_offset + offset:
                                            not_finished = False
                                            break
                                        if i > len(current_window_b) - 1:
                                            current_window_b.append([])
                                            for j in range(start, stop + 1):
                                                current_window_b[-1].append(0.0)
                                        # Read samples_offset amount of samples and add these to the current window
                                        for j in range(start, stop + 1):
                                            current_window_b[i][j - start] = float(line_b[j])
                                        i += 1
                                        line_b = f_b.readline().strip().split(',')
                                        if line_b[0] == '':
                                            not_finished = False
                                            break

                                    # Make sure all windows are of equal length
                                    while i < len(current_window) - 1:
                                        current_window.pop(-1)

                                except EOFError:
                                    not_finished = False

                        # List to keep track of the starting points of all the windows for the final data
                        timestamp_list.append(startpoint)
                        startpoint += offset

                        if not_finished:
                            # Choose the length of the zero padding of the window. Increased definition
                            padding = int(len(current_window) * 6 / 5)
                            # Get the features from the window and the fourier signal for plotting
                            features_time = self.time(current_window, samples_window)
                            ffts, features_fourier = self.fourier(current_window, sampling_frequency,
                                                                  zero_padding=padding - len(current_window),
                                                                  epsilon=epsilon)

                            # Make a list with all the features in the right order per sensor. Right order is first time
                            # features and then frequency
                            features: list[list[float]] = []
                            for i in range(len(features_time)):
                                features.append(features_time[i] + list(features_fourier[i])[1:])

                            # Repeat for b_file
                            if input_file_b != '':
                                padding_b = int(len(current_window_b) * 6 / 5)
                                features_time = self.time(current_window_b, samples_window_b)
                                ffts, features_fourier = self.fourier(current_window_b, sampling_frequency_b,
                                                                      zero_padding=padding_b - len(current_window_b),
                                                                      epsilon=epsilon)
                                for i in range(len(features_time)):
                                    features.append(features_time[i] + list(features_fourier[i])[1:])

                            # Build a string of the feature data. The first element of the string is the timestamp, pop
                            # this timestamp
                            print_timestamps.append(timestamp_list.pop(0))
                            features_list = [str(print_timestamps[-1])]
                            for tup in features:
                                for i, data in enumerate(tup):
                                    features_list.append(str(data))
                            # Add the features to the file if write_to_file is 'y'
                            if write_to_file == 'y':
                                g.write(
                                    str(current_ID + last_index) + ',' + label + ',' + ','.join(features_list) + '\n')

                            # If we want to plot
                            if do_plot:
                                # Mirrored fft for better readability
                                # Frequency axis for non-mirrored fft, account for zero_padding
                                frequency = np.arange(0, sampling_frequency / 2 - 1 / padding,
                                                      sampling_frequency / padding)
                                # Frequency axis for mirrored fft, account for zero_padding
                                # Time axis for the plots
                                time = np.arange(start_offset + offset * k,
                                                 start_offset + offset * k + len(current_window) // sampling_frequency,
                                                 1 / sampling_frequency)

                                # Plotting specs
                                fig, axes = plt.subplots(2, 1)
                                axes[0].plot(time, current_window, label=['X', 'Y', 'Z'])
                                axes[0].legend(loc='upper left')
                                axes[0].set_title("Accelerometer data")
                                axes[1].plot(frequency, abs(ffts[0:round(len(ffts) / 2)]), label=['X', 'Y', 'Z'])
                                axes[1].legend(loc='upper left')
                                axes[1].set_title("Fourier Transform")

                            k += 1
                            current_ID += 1

                    # Writing the first and the last index and the relative path to the video to the output file with
                    # the files that are used.
                    with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt', 'a') as h:
                        h.write(f",{last_index},{last_index + current_ID - 1},{video_file},{video_offset}\n")
                if input_file_b != '':
                    f_b.close()
            plt.show()

            # If do_scale is set to True make a scaled version of the file as well
            if do_scale is True:
                self.SuperStandardScaler(self.output_file)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {input_file_a} at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")

    @staticmethod
    def plot_accelerometer(input_file: str, start_offset: float = 0, stop_offset: float = 0, start: int = 1,
                           stop: int = 3, skip_n_lines_at_start: int = 0) -> None:
        """Function to plot the time-domain curves of data from an input-file.

        Args:
            input_file (str): the relative path of the file with the data, seen from the main file.
            start_offset (float, optional): skip the first r seconds of the data. Defaults to 0.
            stop_offset (float, optional): skip the last r seconds of the data. Defaults to 0.
            start (int, optional): start column from the data file. Defaults to 1.
            stop (int, optional): last column (including) of the data file. Defaults to 3.
            skip_n_lines_at_start (int, optional): skip the first n lines at the start of the data file. Sometimes the first
            few lines contain no data. These lines should be skipped. Defaults to 0.

        Raises:
            FileNotFoundError: error raised if the given input-file cannot be found.
            ValueError: error raised if a data point cannot be converted to a float.
        """

        try:
            # Calculate the sampling frequency and the last timestamp (the third parameter, size, is not used)
            sampling_frequency, last_point = Preprocessing.get_sampling_frequency(input_file, start_offset,
                                                                                  stop_offset)[0:2]

            # Try opening the input-file
            with open(input_file) as f:
                # List with the datapoints, each row will have the data of one sensor
                data: list[list[float]] = []
                # Skip the first 4 lines (contains no data) + the amount of samples in the start_offset
                for _ in range(skip_n_lines_at_start + int(start_offset * sampling_frequency)): f.readline()

                not_finished = True
                while not_finished:
                    line = f.readline().strip().split(',')
                    # The last line of the file is an empty string. When detected we exit the while loop
                    if line[0] == '':
                        not_finished = False
                        break
                    elif float(line[0]) > last_point - stop_offset:
                        not_finished = False
                        break
                    # Read samples_offset amount of samples and add these to the current window
                    data.append([])
                    for j in range(start, stop + 1):
                        data[-1].append(float(line[j]))

        except FileNotFoundError:
            raise FileNotFoundError(f"File {input_file} at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")

        # Define the time axis
        time = np.arange(0, len(data) / sampling_frequency, 1 / sampling_frequency)

        # Plot
        fig, axes = plt.subplots(1, 1)
        axes.plot(time, data)
        plt.show()

    @staticmethod
    def get_time_stats(input_file: str, skip_n_lines_at_start: int = 0) -> tuple[float, float, float, float]:
        """Function to get some stats about the sampling period (average period, standard deviation, minimal and maximal
        period).

        Args:
            input_file (str): the relative path of the file with the data, seen from the main file.
            skip_n_lines_at_start (int, optional): skip the first n lines at the start of the data file. Sometimes the
            first few lines contain no data. These lines should be skipped. Defaults to 0.

        Raises:
            FileNotFoundError: error raised if the given input-file cannot be found.
            ValueError: error raised if a data point cannot be converted to a float.

        Returns:
            tuple[float, float, float, float]: the results: average period, standard deviation, minimal period and
            maximal period
        """

        try:
            # Try opening the file
            with open(input_file) as f:
                # Read the first lines (does not contain data)
                for _ in range(skip_n_lines_at_start):
                    f.readline()

                # Array with the periods
                intervals: np.ndarray[float] = np.array([])
                # First time sample
                t0 = float(f.readline().strip().split(',')[0])
                for line in f:
                    # Checking if the line is not empty (last line is empty)
                    if line != '':
                        # Second time sample
                        t1 = float(line.strip().split(',')[0])
                        # Saving the difference
                        intervals = np.append(intervals, t1 - t0)
                        # Setting the second time sample as the first
                        t0 = t1

            # Calculating the features
            mean = float(np.mean(intervals))
            std = float(np.std(intervals))
            min_inter = min(intervals)
            max_inter = max(intervals)
            return mean, std, min_inter, max_inter

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{input_file}' at the relative path not found!")
        except ValueError:
            raise ValueError(f"FILE CORRUPTED: cannot convert data to float!")

    def SuperStandardScaler(self, input_file: str) -> None:
        """Scale the features with their respective scale. All centroid, peak and total power are put on the same scale.
        By setting do_scale in windowing to True this function is called automatically, else build an object of
        Preprocessing in main and execute Preprocessing.SuperStandardScaler(path).

        Args:
            input_file (str): input file where the unscaled features are stored.
        """

        # First, manually build the data_array, this is because importing with a header and starting columns is a pain
        with open(input_file) as f:
            # Split the set in header, starting columns and actual data
            header = np.array([f.readline().strip().split(',')], dtype='unicode')
            columns = np.array([[0] * 3], dtype='unicode')
            data_array = np.array([[0] * (header.shape[1] - 3)], dtype=float)
            # Fill the data_array
            while True:
                line = f.readline().strip().split(',')
                # Check if the last line was reached
                if line[0] == '':
                    break
                else:
                    # Split the line and add to the two np_arrays
                    columns = np.append(columns, np.array([line[:3]], dtype='unicode'), axis=0)
                    data_array = np.append(data_array, np.array([line[3:]], dtype=float), axis=0)

        # Yeah, append only works when the first element has the same size so I filled one with zeros, sue me
        data_array = data_array[1:]
        columns = columns[1:]

        fa = 0
        if 'acc_x_min' in header:
            fa += 5
        if 'acc_x_pk' in header:
            fa += 3

        # Get amount of features and datapoints
        sensors_amount = (data_array.shape[1]) // fa
        datapoints_amount = data_array.shape[0] - 1

        print(f'Amount of sensors: {sensors_amount}, amount of features per sensor: {fa}')

        if sensors_amount > 6:
            raise ValueError('You have used more than 6 sensors, we have not yet implemented ')
        # If there are more than 3 sensors used, use two different sets.
        # Case < 4 sensors used, max 9 features. Sum 
        for i in range(fa):
            sum_feature = 0
            # Go through every column with the same feature of different sensors and add them
            for j in range(0, min(3, sensors_amount) * fa, fa):
                sum_feature += sum(data_array[:, i + j])
            # Divide to get the mean
            sum_feature = sum_feature / (min(3, sensors_amount) * datapoints_amount)

            # Determine standard deviation of the feature
            std_feature = 0
            for j in range(0, min(3, sensors_amount) * fa, fa):
                for k in range(0, datapoints_amount):
                    std_feature += (data_array[k, i + j] - sum_feature) ** 2
            # Divide by n-1 and take root
            std_feature = (std_feature / (min(3, sensors_amount) * datapoints_amount - 1)) ** 0.5
            # Rescale the columns with their respective feature mean and std
            for j in range(0, min(3, sensors_amount) * fa, fa):
                data_array[:, i + j] = (data_array[:, i + j] - sum_feature) / std_feature
        # If there are more than 3 sensors used, we have gyroscope sensors as well.
        if sensors_amount > 3:
            for i in range(0, fa):
                sum_feature = 0
                # Go through every column with the same feature of different sensors and add them
                for j in range(3 * fa, sensors_amount * fa, fa):
                    sum_feature += sum(data_array[:, i + j])
                # Divide to get the mean
                sum_feature = sum_feature / ((sensors_amount - 3) * datapoints_amount)

                # Determine standard deviation of the feature
                std_feature = 0
                for j in range(3 * fa, sensors_amount * fa, fa):
                    for k in range(0, datapoints_amount):
                        std_feature += (data_array[k, i + j] - sum_feature) ** 2
                # Divide by n-1 and take root
                std_feature = (std_feature / ((sensors_amount - 3) * datapoints_amount - 1)) ** 0.5
                # Rescale the columns with their respective feature mean and std
                for j in range(3 * fa, sensors_amount * fa, fa):
                    data_array[:, i + j] = (data_array[:, i + j] - sum_feature) / std_feature

        # Merge the shit-show
        data_array = data_array.astype('unicode')
        data_array = np.append(columns, data_array, axis=1)
        data_array = np.append(header, data_array, axis=0)

        # Save as csv file
        np.savetxt(f'Preprocessed-data/{self.action_ID}/features_{self.action_ID}_scaled.csv',
                   data_array, fmt='%s', delimiter=',')


def empty_files(files: Iterable[str]) -> None:
    """Function to empty the files given at the input. If a file does not exist, nothing
    is done.

    Args:
        files (Iterable[str]): file locations that will be made empty, seen from the main script
    """

    for file in files:
        # Check whether the file exists
        if os.path.exists(file):
            with open(file, 'w') as f:
                # Write an empty string
                f.write('')
