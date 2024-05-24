from __future__ import annotations
from IPython.display import clear_output, display, HTML
import time
import os
from collections.abc import Collection
from queue import Empty



class VideoLabeler:
    def __init__(self, labels: Collection[str]) -> None:
        self.labels = list(labels)
        self.html_id = 0

    def set_queue(self, queue):
        self.label_input_queue = queue
        
    def labeling(self, video_file: str, timestamp: float, window_size: float,
                 fig_id: int, probs: dict[str, float] = None, process: str = '', video_offset: float = 0,
                 progress: list[int] | None = None) -> str:
        """Function to label a window in a given video at a given timestamp.

        Args:
            video_file (str): relative file-location to the video file.
            timestamp (float): starting point of the window, seen from the start of the video in seconds.
            window_size (float): length of the window in seconds.
            fig_id (int): id of the figure that needs to be displayed next to the video. Should be -1, if there is no
            figure to be displayed.
            probs: (Collection, optional): probability that the frame showed is the corresponding label. Defaults to
            None.
            process: (str, optional): string to print between the video and the prompt. Can be used to indicate the
            process that is executed, e.g. 'TESTING'. Defaults to ''.
            video_offset (float, optional): time in seconds that the video start before the start of the data. Defaults
            to 0.
            progress (list[int], optional): list with the first entry containing the number of the current frame
            and the second entry containing the total amount of frames that need to be labeled. Defaults to None.

        Returns:
            str: the name of the selected label.
        """
        # Clear the output of the cell
        clear_output(wait=True)
        # Making sure that the cell is empty by waiting some time
        # time.sleep(0.1)
        print(timestamp)

      
        # Making sure that the cell is empty by waiting some time

        time.sleep(0.3)
        if process:
            print(process)
        
        """
        This is the location where the label from the Flask app should be retrieved
        This is one of the labels from the buttons on the Active learning HTML page
        """

        while True:
            try:
                # Try to get a label from the queue with a timeout
                new_label = self.label_input_queue.get(timeout=1)

                # If a new label is received from the queue, process it
                if new_label:
                    # Check if the new label is valid
                    if new_label in self.labels or new_label in ['x', 'd', 'r']:
                        return new_label
                    elif new_label == 'n':
                        # Handle the case where a new label is added
                        # You would need to implement how you handle new labels here
                        pass
                    else:
                        print('Invalid label received:', new_label)

            except Empty:
                # No new label was found in the queue, continue with other tasks
                # This is where you can continue with other operations
                # If there's nothing else to do, you can use 'pass' to do nothing
                pass
        


