"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import logging

import cv2
import imutils
import numpy

from image_and_video_prosessors.image_processor import ImageProcessing
from image_and_video_prosessors.videostream_processor import VideoStreamHandler
from report_handlers.csv_report_handler import ReportGenerator
from report_handlers.histogram_handler import HistogramHandler


# pylint: disable=R0903, W0703, I1101
class ObjectRecognition(VideoStreamHandler, ImageProcessing):
    """"
        @brief Class in which is implemented
        logic for object recognition via templates
        matching
    """

    def __init__(self):
        """"
            @brief Class instantiation: setup of class attributes
        """
        try:
            VideoStreamHandler.__init__(self)
            ImageProcessing.__init__(self)
            # private attribute for counting frames in
            # video sequence
            self.__frame_counter: int = 0
            # private attribute for counting detections
            # in video sequence
            self.__detections_counter: int = 0
            # private attribute to be initialized as
            # HistogramHandler object
            self.__histogram_generator: HistogramHandler = None
            #  private attribute initialized as
            # ReportGenerator object
            self.__report_generator: ReportGenerator = ReportGenerator()
        except Exception:
            logging.error("Error occurred during VideoStreamHandler "
                          "object instantiation")

    def __get_frame(self) -> numpy.ndarray:
        """"
            :return numpy.ndarray

            @brief
            Private method in which single frame is grabbed
            from video stream.
        """
        try:
            frame_img: numpy.ndarray = self._video_stream.read()
            frame_resize: numpy.ndarray = imutils.resize(image=frame_img)
            frame_img = ImageProcessing._filter_image(frame_resize)

            return frame_img
        except cv2.error:
            logging.error("Error occurred during getting frame")
            return None

    def __filter_detections(self, detected_object: numpy.int32) -> None:
        """"
            :param detected_object: numpy.int32, object detected in frame
            :return None

            @ brief Private class method in which for а detection is
            get confidence value and based on it is created decision
            box - region of interest in which is detected object and
            information is added in report file
        """
        # creating histogram handler object
        self.__histogram_generator = HistogramHandler()
        # extract the confidence (i.e., probability) associated with
        # the prediction
        if self._detections is not None:
            self._detection_confidence = self._detections[0, 0,
                                                          detected_object, 2]
        else:
            logging.warning("No detections in frame")

        # filter out weak detections by ensurisng the `confidence` is
        # greater than the minimum confidence
        if self._detection_confidence > self._arguments["confidence"]:
            self.__detections_counter += 1
            # extract the index of the class label from detections
            # array, then compute the (x, y)-coordinates of
            # the bounding box for the object
            index: int = int(self._detections[0, 0, detected_object, 1])
            box: numpy.ndarray = self._detections[0, 0, detected_object, 3:7] * \
                                 numpy.array([self._image_width,
                                              self._image_height,
                                              self._image_width,
                                              self._image_height])

            # create prediction frame
            self._create_prediction_frame(box, index)

            # adding record about visualized annotation in report file
            self.__report_generator.add_record(self.__frame_counter,
                                               self._classes_of_interest[index],
                                               self._detection_confidence, box)
            self.__report_generator.close_file()
            self.__histogram_generator. \
                generate_rgb_histogram(self.image, self.__frame_counter)

    def __frame_processing(self) -> None:
        """
            :return None

            @brief
            Private class method in which video stream is
            saved to physical HDD memory while frames are
            processed via calling __filter_detections
            (self, detected_object: numpy.int32) private method.
            Exit from realtime video stream mode is handled.
        """
        # private variable which is used for getting sample
        # frame from video stream
        __sample_frame: numpy.ndarray = self.__get_frame()

        # private variable to be used for saving video
        __video_writer: cv2.VideoWriter = cv2.VideoWriter(
            'results/annotated_video.avi',
            self._fourcc, 20.0, (__sample_frame.shape[1],
                                 __sample_frame.shape[0]))

        if self._detections is None:
            logging.warning("No detections in frame")

        while True:
            # get frame from video stream
            self.image = self.__get_frame()
            if self.image is not None:
                # get object detection classification
                self._get_detections()
                # update frame counter
                self.__frame_counter += 1

                # looping over the detections
                for item in numpy.arange(0, self._detections.shape[2]):
                    self.__filter_detections(item)

                # save output frame in video sequence
                __video_writer.write(self.image)

                # show output frame
                cv2.imshow("Recognized Objects", self.image)
                key: int = cv2.waitKey(1) & 0xFF

                # if the 'Q' key was pressed, break from loop
                if key == ord("q"):
                    break

                # updating the FPS counter
                self._fps.update()

        # releasing video writer stream
        __video_writer.release()

    def real_time_object_recognition(self) -> None:
        """"
            :return None

            @ brief
            Public class method in which is implemented
            object recognition workflow
        """
        # preparing report file and directory
        self.__report_generator.create_results_dir()
        self.__report_generator.create_report()

        # prepare model
        self._prepare_model()

        # starting video stream
        self._start_video_stream()

        # call frame processing method
        self.__frame_processing()

        # stopping video stream
        self._stop_video_stream_and_clean_up()

        # add overview of detected objects in the report
        self.__report_generator.add_report_overview(
            self.__detections_counter,
            self._fps.elapsed(),
            self._fps.fps())

        # closing report file
        self.__report_generator.close_file()
