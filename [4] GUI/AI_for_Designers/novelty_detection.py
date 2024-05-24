from __future__ import annotations

import numpy as np
import pandas as pd
from os import cpu_count

from sklearn.neighbors import LocalOutlierFactor

from IPython.display import HTML, display


class NoveltyDetection:
    def __init__(self, data_file: str, processed_data_files: str) -> None:
        self.data_file = data_file
        self.datapd = pd.read_csv(data_file)
        self.processed_data_files = processed_data_files
        self.html_id = 0

    def detect(self, contamination: int = 0.1) -> list[list[float, str]]:
        """Function that detects anomalies in the data using LocalOutlierFactor.

        Args: 
            contamination (int): The percentage of the dataset that is considered an outlier. Defaults to 0.1.

        Returns:
            list[list[float, str]]: A list containing the lists with the timestamp and the video-file.
        """        

        # Choosing the model LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=contamination, n_jobs=int(cpu_count()*3/4))
        # Fit and predict the model
        prediction = clf.fit_predict(self.datapd.iloc[:, 3:])
        # Counting the outliers, which are represented with the value -1
        count = 0
        for value in prediction:
            if value == -1:
                count += 1
        # Saving a list of the outliers ids  
        ids = np.where(prediction == -1)[0]

        time_video: list[list[float, str]] = []
        for id in ids:
            with open(self.data_file) as f:
                f.readline()
                for _ in range(id):
                    f.readline()
                split = f.readline().strip().split(',')
                i = int(split[0])
                time = float(split[2])
            with open(self.processed_data_files) as f:
                    for line in f:
                        split = line.strip().split(',')
                        if int(split[1]) <= i <= int(split[2]):
                            video = split[3]
                            time_video.append([time, video])
                            break

        return time_video

    def play_novelties(self, time_video: list[list[float | str]], window_size: float) -> None:
        """Function to display the video in the output cell. The video starts automatically at the timestamp,
        plays for window_size seconds and then goes back to the timestamp to loop.

        Args:
            time_video (list[list[float | str]]): list containing a list with a timestamp and a video-file that are
            classified as novelties.
            window_size (float): length of the window in seconds.
        """

        # Add the offset between the start of the video and the data to each timestamp
        video_offset = {}
        with open(self.processed_data_files) as f:
            for line in f:
                split = line.strip().split(',')
                video_offset[split[3]] = float(split[4])

        for time_vid in time_video:
            time_vid[0] += video_offset[time_vid[1]]
              
        # Function to display HTML code  
        display(HTML(f'''
                <head>
                    <script type="text/javascript">
                    let id = 0;
                    const time_video = {time_video};
                    let time = time_video[0][0];
                    const ws = {window_size};

                    function init_nov() {{
                        id = 0;
                        let video = document.getElementById("nov_{self.html_id}");
                        video.currentTime = time;
                        play_nov();
                    }}

                    function play_nov() {{
                        let video = document.getElementById("nov_{self.html_id}");
                        if (video.currentTime < time || video.currentTime >= time + ws) {{
                            video.currentTime = time;
                        }}
                        video.play();
                        setInterval(function () {{
                            if (video.currentTime >= time + ws) {{
                                video.currentTime = time;
                            }}
                        }}, 1);
                    }}

                    function pause_nov() {{
                        let video = document.getElementById("nov_{self.html_id}");
                        video.pause();
                    }}

                    function prev_nov() {{
                    if (id >= 0) {{
                            id = id - 1;
                            time = time_video[id][0];
                            let video = document.getElementById("nov_{self.html_id}");
                            video.setAttribute('src', time_video[id][1]);
                            document.getElementById("content").innerHTML = 'Novelty ' + (id + 1) + ' out of {len(time_video)} at ' + time + 's in ' + time_video[id][1];
                            play_nov();
                        }}
                    }}

                    function next_nov() {{
                        if (id < {len(time_video)} - 1) {{
                            id = id + 1;
                            time = time_video[id][0];
                            let video = document.getElementById("nov_{self.html_id}");
                            video.setAttribute('src', time_video[id][1]);
                            document.getElementById("content").innerHTML = 'Novelty ' + (id + 1) + ' out of {len(time_video)} at ' + time + 's in ' + time_video[id][1];
                            play_nov();
                        }}
                    }}

                    </script>
                    <title></title>
                </head>
                <body>
                    <video id="nov_{self.html_id}" width="500px" src="{time_video[0][1]}" muted></video>
                    <br>
                    <script type="text/javascript">init_nov()</script>
                    <button onClick="play_nov()">Play</button>
                    <button onClick="pause_nov()">Pause</button>
                    <button onClick="prev_nov()">Previous novelty</button>
                    <button onClick="next_nov()">Next novelty</button>
                    <div id="content">Novelty 1 out of {len(time_video)} at {time_video[0][0]}s in {time_video[0][1]}</div>
                </body>
        '''))

        self.html_id += 1
