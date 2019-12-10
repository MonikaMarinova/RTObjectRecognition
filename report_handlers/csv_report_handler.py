"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import io
import logging
import os

import imutils.video
import numpy


# pylint: disable=W0703, I1101
class ReportGenerator:
    """"
        @brief Class in which is implemented logic for
        creating results and histograms directories, run
        report text file is generated
    """

    def __init__(self):
        """"
            @brief Class instantiation: setup of class attributes
        """
        try:
            # private class attribute project directory path holder
            self.__project_root_dir: str = os.getcwd()
            # private class attribute results directory path holder
            self.__results_dir: str = self.__project_root_dir + '\\results'
            # private class attribute to be used as report file holder
            self.__report_file: io.TextIOWrapper = None
        except Exception:
            logging.error("Error occurred during ReportGenerator "
                          "object instantiation")

    def create_results_dir(self) -> None:
        """"
            :return None

            @brief Public class method in which work directory is changed
            and results directory is created if doesn't exists
        """
        try:
            # changing work directory
            os.chdir(self.__project_root_dir)
            # create results directory if it is not existing
            if not os.path.isdir('results'):
                os.mkdir('results')
            self.__create_histogram_dir()
        except PermissionError:
            logging.error("Permission error occurred during "
                          "creating results directory")

    def __create_histogram_dir(self) -> None:
        """"
            :return None

            @brief Private class method for creating
            histograms plots directory
        """
        os.chdir(self.__results_dir)
        try:
            if not os.path.isdir('histograms'):
                os.mkdir('histograms')
        except PermissionError:
            logging.error("Permission error occurred during "
                          "creating histograms directory")

        os.chdir(self.__project_root_dir)

    def create_report(self) -> None:
        """"
            :return None

            @brief Public class method in which report file is created,
            header row is writen in it
        """
        # setting header row and file_name formats
        header_row: str = 'Report_format: \n' \
                          'frame_number;object_type;detection' \
                          '_confidence' \
                          ';region_of_interest;coordinates \n \n'
        file_name: str = 'results\\annotation_report'
        try:
            # open report file, mode: write only
            with open(file_name, 'w') as self.__report_file:
                # write first row - header row
                self.__report_file.write(header_row)
        except PermissionError:
            logging.error("Permission error occurred during "
                          "creating report file")
        except IOError:
            logging.error("IOError occurred during writing into report file")

    def add_record(self, frame_number: int, obj_type: str,
                   detection_confidence: float,
                   coordinates_arr: numpy.ndarray) -> None:
        """"
            :param frame_number: int, frame number
            :param obj_type: str, object classification type
            :param detection_confidence: float, object classification accuracy
            between 0 and 1
            :param coordinates_arr: numpy.ndarray, array of four coordinates
            which are region of interest for the detected object

            :return None

            @brief Public class method in which row with information about
            one detected object is added
        """
        # setting row record format and file name
        if frame_number <= 9:
            frame_number = '0' + str(frame_number)

        row: str = str(frame_number) + ';' + obj_type + ';' + \
              str(detection_confidence) + ';' + \
              str(coordinates_arr) + '\n'

        file_name: str = 'results\\annotation_report'

        try:
            # open report file, mode: append text
            with open(file_name, 'a') as self.__report_file:
                # write a row
                self.__report_file.write(row)
        except PermissionError:
            logging.error("Permission error occurred during "
                          "creating report file")
        except IOError:
            logging.error("IOError occurred during writing into report file")

    def add_report_overview(self, detections_cnt: int,
                            elapsed_time: imutils.video.fps.FPS,
                            approximate_fps: imutils.video.fps.FPS) \
            -> None:
        """"
             :param detections_cnt: int, total count of detected objects
             during recording
             :param elapsed_time: imutils.video.fps.FPS, total time
             for recording
             :param approximate_fps: imutils.video.fps.FPS, average frames
             per second rate

             :return None

             @brief Public class method in which rows with information
             about total count of detected objects during recording,
             total time for recording and average frames
             per second rate in the end of report file
         """
        # setting records format and file name
        detections_count_row: str = "Number of detected object: " + \
                                    str(detections_cnt) + '\n'
        elapsed_time_row: str = "Elapsed time: " + \
                                str(elapsed_time) + '\n'
        approximate_fps_row: str = "Approximate FPS: " + \
                                   str(approximate_fps) + '\n'
        file_name: str = 'results\\annotation_report'

        try:
            # open report file, mode: append text
            with open(file_name, 'a') as self.__report_file:
                # writing over view rows
                self.__report_file.write(detections_count_row)
                self.__report_file.write(elapsed_time_row)
                self.__report_file.write(approximate_fps_row)
        except PermissionError:
            logging.error("Permission error occurred during "
                          "creating report file")
        except IOError:
            logging.error("IOError occurred during writing into report file")

    def close_file(self) -> None:
        """"
            :return None

            @brief Public class method for closing
            private file
        """
        self.__report_file.close()
