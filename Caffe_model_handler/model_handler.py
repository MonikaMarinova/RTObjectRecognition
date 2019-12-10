"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import argparse
import logging

import cv2


# pylint: disable=R0903, W0703, I1101
class CaffeModelHandler:
    """"
        @brief Class in which is implemented logic
        for working with pre-trained models data base
    """

    def __init__(self):
        """
            @brief Class instantiation: setup of class attributes
        """
        try:
            # private attribute to be used as argument parser
            self.__arg_parser: argparse.ArgumentParser = argparse. \
                ArgumentParser()
            # protected attribute of type dict, in which are stored all command
            # line arguments that are already added
            self._arguments: dict = []
            # protected attribute in which will be loaded serialized
            # model from disk
            self._net: cv2.dnn_Net = cv2.dnn_Net()
            # protected attribute which initializes list of class labels
            # MobileNet was trained to detect
            self._classes_of_interest: list = ["background", "aeroplane",
                                               "bicycle", "bird", "boat",
                                               "bottle", "bus", "car", "cat",
                                               "chair", "cow", "diningtable",
                                               "dog", "horse", "motorbike",
                                               "person", "pottedplant",
                                               "sheep", "sofa", "train",
                                               "tvmonitor"]
        except Exception:
            logging.error("Error occurred during CaffeModelHandler"
                          " object instantiation")

    def _add_parsers(self) -> None:
        """
            :return None

            @brief
            Protected class method in which command line arguments are added and
            parsed. They are added to arguments dictionary as well.
        """
        try:
            # constructing the argument parse and parse arguments
            self.__arg_parser: argparse.ArgumentParser = argparse. \
                ArgumentParser()
            self.__arg_parser.add_argument('-p', '--prototxt',
                                           help="path to Caffe 'deploy'"
                                                " prototxt file",
                                           default="MobileNetSSD_deploy."
                                                   "prototxt.txt")
            self.__arg_parser.add_argument('-m', '--model',
                                           help="path to Caffe pre-trained "
                                                "model",
                                           default="MobileNetSSD_deploy."
                                                   "caffemodel")
            self.__arg_parser.add_argument('-c', '--confidence', type=float,
                                           default=0.2, help="minimum"
                                                             " probability"
                                                             " to filter weak "
                                                             "detections")
            self.__arg_parser.add_argument('-s', '--source',
                                           help="Source of video stream "
                                                "(webcam/host)")
            self._arguments = vars(self.__arg_parser.parse_args())
        except Exception:
            logging.error("Error occurred during parsing arguments")

    def _load_serial_model(self) -> bool:
        """
            :return bool

            @brief
            Protected class method in which is loaded serialized model
            from disk.
        """
        print("[INFO] Loading model...")
        try:
            self._net: cv2.dnn_Net = cv2.dnn.readNetFromCaffe \
                (self._arguments["prototxt"], self._arguments["model"])
            return True
        except cv2.error:
            logging.error("Error occurred during loading serial model")
            return False
