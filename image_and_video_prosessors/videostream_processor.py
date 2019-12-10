"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import logging
import time

import cv2
import imutils
import imutils.video


# pylint: disable=R0903, W0703, I1101
class VideoStreamHandler:
    """"
        @brief Class in which is implemented logic for handling
        real time video stream - starting and stopping it
    """

    def __init__(self):
        """"
            @brief Class instantiation: setup of class attributes
        """
        try:
            # protected attribute which will be used for starting video stream
            self._video_stream: imutils.video.webcamvideostream. \
                WebcamVideoStream \
                = imutils.video.webcamvideostream.WebcamVideoStream()
            # protected attribute to be used for FPS counter
            self._fps: imutils.video.fps.FPS = imutils.video.fps.FPS()
            # protected attribute to be used as video codec format holder
            self._fourcc: cv2.VideoWriter_fourcc = cv2. \
                VideoWriter_fourcc(*'XVID')
        except Exception:
            logging.error("Error occurred during VideoStreamHandler"
                          " object instantiation")

    def _start_video_stream(self) -> None:
        """
            :return None

            @brief
            Protected class method  in which is initialized real time video
            stream, allowing the camera sensor to warm up and is
             initialize the FPS counter.
        """
        logging.info("Starting camera...")
        try:
            self._video_stream: imutils.video.webcamvideostream. \
                WebcamVideoStream = imutils.video.VideoStream(src=0).start()
            time.sleep(2.0)
            self._fps: imutils.video.fps.FPS = imutils.video.fps.FPS().start()
        except Exception:
            logging.error("Error occurred during starting web camera")

    def _stop_video_stream_and_clean_up(self) -> None:
        """
            :return None

            @brief
            Protected class method in which video stream
            is stopped and the window is cleaned up.
        """
        try:
            # stopping timer and display FPS information
            self._fps.stop()
            logging.info("Elapsed time: {:.2f}".format(self._fps.elapsed()))
            logging.info("Approximate FPS: {:.2f}".format(self._fps.fps()))

            # clean up
            self._video_stream.stop()
            cv2.destroyAllWindows()
        except Exception:
            logging.error("Error occurred during stopping web camera")
