from __future__ import annotations

from AI_for_Designers.Videolabeler import VideoLabeler

import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from collections.abc import Sequence
from typing import Any, Set


class ActiveLearning:
    """Class to use active learning to predict what action is executed in each window"""

    def __init__(self, data_file: str, activity: str, labels: list[str], window_size: float) -> None:
        self.pca = None
        self.preds: np.ndarray | None = None
        self.unpreds: np.ndarray | None = None
        self.static_folder = 'static'

        random.seed(42)
        # Get the data and store in a pandas dataframe
        self.data_file = data_file
        self.datapd = self.get_sensor_data(data_file)
        # Get the amount of features by removing the ID, label and timestamp
        self.number_of_features = self.datapd.shape[1] - 3
        # Determine model in separate function
        self.model = self.determine_model()
        # PCA
        self.determine_pca()
        # Name of the activity
        self.action_ID = activity
        # Last ID for redo button
        self.lastID = -1
        # Argument is the set of labels that the user predicts
        self.labels = labels
        # A list of the ID's that we have labeled already
        self.labeled_ids = []
        # This is for measuring the functionality/efficiency of the Active learning
        self.gini_margin_acc: list[list[float]] = []

        # Attributes for the video labeler
        self.vid = VideoLabeler(labels)
        self.window_size = window_size
        self.html_id = -1

        # X_pool is the dataset that we use for building the model. X_test is to test the model
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.split_pool_test()
        self.X_test = np.array(self.X_test)
        # Remove the labels from the X_pool set
        self.X_pool['label'] = [''] * self.X_pool.shape[0]

    def set_queue(self, queue):
        """Sets the queue for the VideoLabeler instance."""
        self.vid.set_queue(queue)

    @property
    def unlabeled_ids(self) -> set[int]:
        """We make a property so that when the list of labeled_ids changes, we don't have to worry about changing
        this one."""
        return set(range(self.X_pool.shape[0])) - set(self.labeled_ids)

    @staticmethod
    def determine_model(max_depth: int | None = None) -> RandomForestClassifier:
        """Return the selected model for this action classification.

        Args:
            max_depth (int, optional): maximum depth of the chosen model. Defaults to None.

        Returns:
            Object: returns the model that we want to use for this action classification.
        """

        return RandomForestClassifier(max_depth=max_depth, criterion='gini')

    def update_model(self) -> None:
        """This model determines the current average max depth of the trees in the random forest. If the depth has
        changed drastically since the last check we update the model"""
        # Determine the depth of the current model
        forest = self.determine_model().fit(self.preds[:, 3:], self.preds[:, 1])
        # Multiplication factor of 1.25 so that the tree can grow while actively training but has a limit to prevent
        # overfitting
        avg_depth = int(sum(estimator.tree_.max_depth for estimator in forest.estimators_) / 100 * 1.25) + 1
        self.model = self.determine_model(avg_depth)

    @staticmethod
    def get_sensor_data(data_file: str) -> pd.DataFrame:
        """Read and return the datafile from the given path.

        Args:
            data_file (str): location of the datafile (csv).

        Returns:
            pd.dataframe: pandas dataframe of the datafile.
        """
        return pd.read_csv(data_file)

    def split_pool_test(self) -> list[pd.DataFrame]:
        """Splits a dataset into a pool and a test set.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: returns the X_pool, X_test, y_pool, y_test.
        """
        random_state = 42
        test_size = 0.2
        return train_test_split(self.datapd, self.datapd['label'], test_size=test_size, random_state=random_state)

    def training(self, maximum_iterations, random_points: int = 4, cluster_points: int = 1) -> list[str]:
        """The process of training the datapoints, first set starting points, then iterate until you have a certainty.

        Args:
            maximum_iterations (_type_): Maximum amount of iterations.
            random_points (int, optional): Number of random starting points. Defaults to 3.
            cluster_points (int, optional): Number of clustered starting points. Defaults to 1.
        """

        # Remove prediction point images to prevent flooding when you stop active learning prematurely
        self.remove_pngs()
        # Set randomized starting points       
        self.set_starting_points(random_points)

        # Set the predicted and und predicted sets into new arrays, these will be used further on
        self.preds = np.array(self.X_pool.loc[self.X_pool['label'] != ''])
        self.unpreds = np.array(self.X_pool.loc[self.X_pool['label'] == ''])

        self.clustered_starting_points(cluster_points)

        self.update_model()

        # Set the most ambiguous points iteratively
        self.iterate(maximum_iterations)

        # Ensure the Models directory exists
        os.makedirs('Models', exist_ok=True)

        # Save the model as a pickle file
        with open(fr'Models/model_{self.action_ID}_{maximum_iterations}.pickle', 'wb') as f:
            pickle.dump(self.model, f)

        # Return the labels, you may find new labels while training
        return self.labels

    def set_starting_points(self, n_samples: int) -> None:
        """Generates a training set by selecting random starting points, labeling them, and checking if there's an
        instance of every activity.

        Args:
            n_samples (int): Amount of random samples that you want active learning to start with. This is a
            multiplication factor so you n_samples = 3 * 4 labels = 12 sample points.
        """
        # Keep track of what activities we have labeled already
        seen_activities = []  # list of strings
        # Amount of datapoints that we randomly sample
        range_var = n_samples * len(self.labels)
        # Generate random points
        for _ in range(range_var):
            # Pick a random point from X_pool
            while True:
                # Set a random id that is in the X_pool and has not yet been labeled
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id not in self.labeled_ids and random_id in self.X_pool['ID']:
                    break
            # Give the timestamp to the identification module but for testing I have automated it
            got_labeled = self.identify(random_id)
            if got_labeled == 'x':
                self.datapd.drop(random_id, 0)
                self.X_pool.drop(random_id, 0)
            # If this label was not accounted for we add it to the set of labels
            else:
                # Redo button:
                if got_labeled == 'r':
                    try:
                        # Remove it from the list that we will identify
                        del self.labeled_ids[-1]
                        del seen_activities[-1]
                        got_labeled = self.identify(random_id)
                    # Catch when you remove the first element
                    except IndexError:
                        raise ValueError("You ain't nah removin' nottin")
                    if got_labeled == 'x':
                        self.datapd.drop(random_id, 0)
                        self.X_pool.drop(random_id, 0)
                        continue
                # Add it to the labels list if it is a new label
                if not (got_labeled in self.labels or got_labeled == 'r'):
                    self.labels.append(got_labeled)
                seen_activities.append(got_labeled)
                self.labeled_ids.append(random_id)
                self.lastID = random_id
        # Fill the X_pool
        for i in range(len(self.labeled_ids)):
            # print(np.where(self.labeled_ids[i]), self.labeled_ids[i])
            self.X_pool.at[self.labeled_ids[i], 'label'] = seen_activities[i]

    def clustered_starting_points(self, n_samples: int = 1) -> None:
        """Find the training point based on a clustering algorithm. You should be able to find at least one sample of
        each label you predict will be in the dataset.

        Args:
            n_samples (int): Amount of samples that you want from each cluster centre. Keep this value at 1, it is not
            properly tested. TODO
        """
        # Amount of clusters that we expect
        # There is a multiplier of 1.5 so that small clusters are not overlooked
        n_clusters = int(len(self.labels) * 1.5) + 1
        # Determine the means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.unpreds[:, 3:])
        cert_indices = []
        predictions = np.array(kmeans.predict(self.unpreds[:, 3:]))
        X = self.unpreds.copy()
        X[:, 1] = predictions
        for label in range(n_clusters):
            # Select samples with label
            x = X[np.where(X[:, 1] == label)[0], :]
            for _ in range(n_samples):  # For now
                # Transform
                total_dists = np.sum(kmeans.transform(x[:, 3:]) ** 2, axis=1)
                # Add certain samples
                cert_indices.append(x[np.argmin(total_dists), 0])
                # print(cert_indices)
                x = np.delete(x, np.argmin(total_dists), 0)
        for e in cert_indices:
            got_labeled = self.identify(e)  # for testing
            if got_labeled == 'x':
                self.remove_row(e)
            else:
                # Redo button
                if got_labeled == 'r':
                    self.preds = np.delete(self.preds, np.where(self.preds[:, 0] == self.lastID), 0)
                    got_labeled = self.identify(e)
                    if got_labeled == 'x':
                        self.remove_row(e)
                        continue
                    elif got_labeled == 'r':
                        raise ValueError('You sneaky foo. please stop trying to break our code. As punishment you shall'
                                         'be labeling from the start')
                # Add the label to the label list if it is a new label
                if not (got_labeled in self.labels or got_labeled == 'r'):
                    self.labels.append(got_labeled)
                self.labeled_ids.append(e)
                line = self.X_pool.iloc[np.where(self.X_pool.iloc[:, 0] == e)[0][0], :].copy()
                line.at['label'] = got_labeled
                line = np.array(line).reshape(1, -1)
                self.preds = np.append(self.preds, line, axis=0)
                self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == e), 0)
                self.lastID = e

    def iterate(self, max_iter: int) -> None:
        """This function is the iterative process of active learning. Labeling the most ambiguous points.

        Args:
            max_iter (int): maximum number of iterations.
        """

        # This function is the iterative process of active learning. Labeling the most ambiguous points
        iter_num = 0
        while True:
            iter_num += 1
            # find most ambiguous point (find_most_ambiguous_id)
            # label it (set_ambiguous_point)
            # add to training data 
            new_index, margin = self.set_ambiguous_point([iter_num, max_iter])
            # Iterate for a decided number of points or until you have a certain margin
            if iter_num >= max_iter or margin > 0.15:
                
                break

    def set_ambiguous_point(self, progress: list[int] | None = None) -> tuple[int, int]:
        """Lets the designer label an ambiguous point.

        Args:
            progress (list[int], optional): list with the first entry containing the number of the current iteration
            and the second entry containing total iterations. Defaults to None.

        Returns:
            tuple[int, int]: return the ID that has been labeled and the margin (certainty) of the point that has
            been labeled.
        """

        self.html_id = time.time()
        self.remove_pngs()
        # Determine the ID of the most ambiguous datapoint      
        get_id_to_label, margin, les_probs = self.find_most_ambiguous_id()
        # Add it to the IDs that we have labeled
        self.labeled_ids.append(get_id_to_label)
        # Print PCA
        self.print_prediction_point(get_id_to_label)
        # Get what label this ID is supposed to get
        # Just for testing, add les_probs as arg to les_probs if you want these to be printed
        new_label = self.identify(get_id_to_label, les_probs=les_probs, progress=progress)
        if new_label == 'x':
            self.remove_row(get_id_to_label)
            return get_id_to_label, 0
        else:
            # Redo button
            if new_label == 'r':
                self.preds = np.delete(self.preds, np.where(self.preds[:, 0] == self.lastID), 0)
                new_label = self.identify(get_id_to_label, les_probs=les_probs)

                if new_label == 'x':
                    self.remove_row(get_id_to_label)
                    return get_id_to_label, 0
                elif new_label == 'r':
                    raise ValueError('You sneaky foo. please stop trying to break our code. As punishment you shall be'
                                     'labeling from the start')
            if not (new_label in self.labels or new_label == 'r'):
                self.labels.append(new_label)
            # Extract the row from the unpredicted array
            t = self.unpreds[self.unpreds[:, 0] == get_id_to_label, :]
            # Label the row
            t[0, 1] = new_label
            # Stack it onto the predicted array
            self.preds = np.vstack((self.preds, t))
            # Delete it from the unpredicted array
            self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == get_id_to_label)[0][0], 0)

            self.lastID = get_id_to_label
            if get_id_to_label in self.unpreds[:, 1]:
                raise ValueError('you did an oopsie')

            # Return the label and the margin

            self.write_margin_to_file(new_label, margin * 100)  # Pass margin as percentage

            return get_id_to_label, margin
        
    def write_margin_to_file(self, label, margin):
        """Writes the label and its certainty margin to a file in the 'session_data' directory."""
        file_path = "session_data/label_certainty.txt"  # Adjust the path to the correct folder
        with open(file_path, 'w') as file:  # 'w' mode to overwrite each time
            file.write(f'{label}: {margin:.2f}%\n')

      

    def identify(self, id, les_probs: dict[str, float] | None = None, process: str = '',
                 progress: list[int] | None = None) -> str:
        """This function will call the identification system from Gijs en Timo, for now it has been automated.

        Args:
            id (int): the ID that will be labeled.
            les_probs (dict[str, float], optional): when in iterate, this tuple has the probabilities of the current
            datapoint to each label. When defaulted to None this does not print. Defaults to None.
            process: value should be 'TESTING' if this is the case and empty otherwise. Value will be printed below
            the video and above the prompt. Defaults to ''.
            progress (list[int], optional): list with the first entry containing the number of the current iteration
            and the second entry containing total iterations. Defaults to None.

        Returns:
            str: return the return of the video labeler, so the class that the point is labeled as.
        """
        
        def write_timestamp_to_file(timestamp):
            file_path="session_data/timestamp.txt"
            with open(file_path, "w") as file:
                file.write(str(timestamp))

        timestamp = self.datapd.iloc[id, 2]
        write_timestamp_to_file(timestamp)
        
        with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt') as f:
            for line in f:
                split = line.strip().split(',')
                if int(split[1]) <= id <= int(split[2]):
                    video_file = split[3]
                    video_offset = float(split[4])
                    break

        if les_probs is None:
            return self.vid.labeling(video_file, timestamp, self.window_size, self.html_id, process=process,
                                     video_offset=video_offset, progress=progress)
        else:
            return self.vid.labeling(video_file, timestamp, self.window_size, self.html_id, les_probs, process=process,
                                     video_offset=video_offset, progress=progress)

    def find_most_ambiguous_id(self) -> tuple[int, float, dict[str, float]]:
        """Finds the most ambiguous sample. The unlabeled sample with the greatest difference between most and second
        most probably classes is the most ambiguous.

        Raises:
            ValueError: exception for testing purposes.

        Returns:
            tuple[int, float, dict[str, float]]: returns the id of the most ambiguous sample, the margin and the
            probabilities.
        """

        try:
            # Fit the model with the datapoints that we have currently labeled.
            self.model.fit(self.preds[:, 3:], self.preds[:, 1])
            # Use this model to get probabilities of datapoints belonging to a certain class.
            sorted_preds = np.sort(self.model.predict_proba(self.unpreds[:, 3:]), axis=1)
            # Basses for the lowest margins
            lowest_margin = 2.0
            lowest_margin_sample_id: int = 0
            # Append an empty list for the results of this iteration
            self.gini_margin_acc.append([0., 0., 0.])
            # Make a list of the unlabeled ids and sort it
            unlbld = list(self.unlabeled_ids)
            unlbld.sort()
            
            # Iterate for the length of datapoints that you have not yet labeled
            for i in range(sorted_preds.shape[0]):
                # Subtract from the most certain class the second to most certain class
                margin = sorted_preds[i, -1] - sorted_preds[i, -2]
                # Is it the lowest?
                if margin < lowest_margin:
                    lowest_margin_sample_id = self.unpreds[i, 0]
                    lowest_margin = margin
                # Add the gini of the datapoint to gini of this iteration
                self.gini_margin_acc[-1][0] += self.gini_impurity_index(list(sorted_preds[i, :]))
            # Make it an average and add the lowest margin
            self.gini_margin_acc[-1][0] /= len(unlbld)
            self.gini_margin_acc[-1][1] = lowest_margin

            les_probs: dict[str, float] = {}
            for label, prob in zip(self.model.classes_, self.model.predict_proba(
                    self.unpreds[np.where(self.unpreds[:, 0] == lowest_margin_sample_id)[0], 3:]).tolist()[0]):
                les_probs[label] = prob
            # Oeh fun result get better with more samples Oeh!
            return lowest_margin_sample_id, lowest_margin, les_probs
        # Exception mostly for testing idk if it will every be handydany again
        except ValueError:
            raise ValueError(self.preds)

    @staticmethod
    def gini_impurity_index(list_of_p) -> float:
        """Returns the gini: 1 - sum(p^2).

        Args:
            list_of_p (list[float]): A list of the probabilities of the to be labeled point belongs to each class.

        Returns:
            float: the gini impurity index, to be used for of evaluation your model.
        """

        # Return the gini: 1 - sum(p^2)
        return 1 - sum((item * item for item in list_of_p))

    def write_to_file(self) -> None:
        """Make a prediction of the unpredicted dataset, fill the predicted set with these predictions. sort the array on
        index and write it to a _AL_predictionss.csv file."""

        self.unpreds[:, 1] = self.model.predict(self.unpreds[:, 3:])
        nptofile = np.append(self.preds, self.unpreds, axis=0)
        nptofile = nptofile[nptofile[:, 0].argsort()]
        # print(self.preds[:5, :])
        output = fr"{self.data_file.split('.csv')[0]}_AL_predictions.csv"
        names = np.array([self.datapd.columns])
        np.savetxt(output, np.append(names, nptofile, axis=0), delimiter=",", fmt='%s')
        print(f'Predictions written to {output}')

    def testing(self, n_to_check: int | None = None) -> None:
        """Checks for overwriting based on randomized sampling. To avoid having to make them label the entire test set,
        we ask the designer to confirm n predicted test labels.

        Args:
            n_to_check (int | None, optional): Amount of values that you want tested. When None is given, 
            you will iterate through the entire test set (20% of the entire sample size).

        Returns:
            int: error_count and n_to_check
        """

        self.html_id = -1
        # Check for None or numerical size
        if n_to_check is None or n_to_check > len(self.X_test):
            n_to_check = len(self.X_test)
            test_ids = []
            # j is to remember how many samples you deleted
            k = 0
            # Find amount of values that you still need
            while len(test_ids) != n_to_check:
                random_id = random.randint(0, self.datapd.shape[0])
                # Find testing ids
                if random_id in self.X_test[:, 0] and random_id not in test_ids:
                    test_ids.append(random_id)
            predictions = self.model.predict(np.array(self.datapd.iloc[test_ids, 3:]))
            error_count = 0

            # Iterate through the test ids
            for j in range(len(test_ids)):
                result = self.identify(test_ids[j], process='TESTING')
                if result == 'x':
                    k += 1
                elif predictions[j] != result:
                    error_count += 1

            print(f'Error rate: {error_count / (n_to_check - k)} ({n_to_check - k} samples)')

        else:
            # Make sure that 
            i = 0
            # Make sure you always test the amount of values that you gave with n_to_check, even if you delete some
            # sample
            while i < n_to_check - 1:
                test_ids = []
                # Find amount of values that you still need, even if samples have been deleted
                while len(test_ids) != n_to_check - i:
                    random_id = random.randint(0, self.datapd.shape[0])
                    # Find testing ids
                    if random_id in self.X_test[:, 0] and random_id not in test_ids:
                        test_ids.append(random_id)
                predictions = self.model.predict(np.array(self.datapd.iloc[test_ids, 3:]))
                error_count = 0

                # Iterate through the test ids
                for j in range(len(test_ids)):
                    result = self.identify(test_ids[j], process='TESTING')
                    if result == 'x':
                        pass
                    elif predictions[j] != result:
                        error_count += 1
                        i += 1
                    else:
                        i += 1
            print(f'Error rate: {error_count / n_to_check} ({n_to_check} samples)')

    def remove_row(self, id: int) -> None:
        """Removes a row from the data.

        Args:
            id (int): id of the row to be removed.
        """

        self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == id)[0][0], 0)
        self.datapd.drop(id, 0)

    def determine_pca(self) -> None:
        """Calculates and saves the pca of the data in self.pca."""

        pca = PCA(n_components=2, svd_solver='auto')
        self.pca = np.array(pca.fit_transform(self.datapd.iloc[:, 3:]))
        self.pca = np.append(np.array([[i for i in range(len(self.datapd))]]).reshape(-1, 1), self.pca, axis=1)

    @staticmethod
    def remove_pngs() -> None:
        directory = r'Plots'
        if os.path.exists(directory) and os.path.isdir(directory):
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f) and 'plot_to_label' in f:
                    os.remove(f)

    def print_prediction_point(self, current_id: int) -> None:
        plt.clf()
        plt.scatter(self.pca[:, 1], self.pca[:, 2], c='grey')
        for label in self.labels:
            lst = self.preds[np.where(self.preds[:, 1] == label)[0], :]
            lst = lst[:, 0].tolist()
            temp_pca = [e[1:].tolist() for e in self.pca if int(e[0]) in lst]

            x, y = zip(*temp_pca)
            plt.scatter(x, y, label=label)
        
        static_plots_folder = os.path.join('session_data', 'plots')
        if not os.path.exists(static_plots_folder):
            os.makedirs(static_plots_folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = f'plot_to_label_{self.html_id}_{timestamp}.png'
        plot_filepath = os.path.join(static_plots_folder, plot_filename)

        plt.savefig(plot_filepath)
        print(f"Plot saved to: {plot_filepath}")

    def plotting(self) -> None:
        """Plot the gini index, the margin, and the test accuracy on every iteration."""
        plt.clf()
        plt.plot(np.array(self.gini_margin_acc)[:, :2], label=['Gini index', 'Margin'])
        plt.xlabel('Iterations [n]')
        plt.ylabel('Uncertainty')
        plt.title('Active Learning Performance')
        plt.legend()

        # Consider saving this plot if needed, similar to `print_prediction_point`
        # plt.savefig(plot_filepath)
    