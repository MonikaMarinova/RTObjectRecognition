"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import logging

import cv2
import numpy

from Caffe_model_handler.model_handler import CaffeModelHandler


# pylint: disable=R0903, W0703, I1101
class ImageProcessing(CaffeModelHandler):
    """"
        @brief Class in which is implemented logic for
        static image processing, detections are found and
        classified vie template matching
    """

    def __init__(self):
        """"
            @brief Class instantiation: setup of class attributes
        """
        try:
            CaffeModelHandler.__init__(self)
            # private attribute which generates a set of bounding box
            # colors for each class
            self.__colors: numpy.ndarray = numpy.random.uniform \
                (0, 255, size=(len(self._classes_of_interest), 3))
            # protected attribute to be used as detections array holder
            self._detections: numpy.ndarray = None
            # protected attribute to be used az detection confidence holder
            self._detection_confidence: numpy.float32 = None
            # public attribute to be used as frame image holder
            self.image: numpy.ndarray = None
            # protected attribute for frame height
            self._image_height: int = None
            # protected attribute for frame width
            self._image_width: int = None
        except Exception:
            logging.error("Error occurred during ImageProcessing"
                          " object instantiation")

    def _prepare_model(self) -> None:
        """"
            :return None

            @brief
            Protected class method in which parsers are added
            and model is loaded. For both are used CaffeModelHandler
            methods.
        """
        self._add_parsers()
        if self._load_serial_model():
            logging.info("Model is loaded!")

    def __image_to_blob(self) -> numpy.ndarray:
        """"
            :return numpy.ndarray

            @brief
            Private method in which the frame dimensions are get
            and converted into a blob
        """
        (self._image_height, self._image_width) = self.image.shape[:2]
        try:
            src_blob: numpy.ndarray = cv2.dnn.blobFromImage(
                cv2.resize(src=self.image, dsize=(300, 300)),
                scalefactor=0.007843, size=(300, 300), mean=127.5)

            return src_blob
        except cv2.error:
            logging.error("Error occurred during converting image to blob")
            return None

    def _get_detections(self) -> numpy.ndarray:
        """"
            :return numpy.ndarray

            @ brief
            Protected method in which blob is passed through the network
            and the detections and predictions are obtained
        """
        try:
            if self.__image_to_blob() is not None:
                blob: numpy.ndarray = self.__image_to_blob()
                self._net.setInput(blob)
                self._detections = self._net.forward()

            return self._detections
        except Exception:
            logging.error("Error occurred during getting detections")

    def _create_prediction_frame(self, src_box: numpy.ndarray, index: int) \
            -> None:
        """"
            :param src_box: numpy.ndarray, bounding box for detected object ;
            :param index: int, index of value in self.detections array
            :return None

            @brief
            Protected class method in which detected object is labeled
            and rectangle frame for it is visualized.
        """
        # unpacking both four  vertices of a square box
        # start_x, start_y, end_x, end_y are of type numpy.int32
        (start_x, start_y, end_x, end_y) = src_box.astype("int")

        # labeling
        label: str = "{}: {:.2f}%".format(
            self._classes_of_interest[index], self._detection_confidence)

        try:
            # drawing rectangle for detection frame
            cv2.rectangle(img=self.image, pt1=(start_x, start_y),
                          pt2=(end_x, end_y), color=self.__colors[index],
                          thickness=2)

            # variable used for correcting the y coordinate
            y_coordinate: numpy.int32 = ImageProcessing \
                .__coordinate_correction(numpy.int32(15), start_y)

            # visualizing label string as text on the frame
            cv2.putText(img=self.image, text=label,
                        org=(start_x, y_coordinate),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=self.__colors[index], thickness=2)
        except cv2.error:
            logging.error("Error occurred during creating region of interest "
                          "frame borders ")

    @staticmethod
    def _filter_image(input_img: numpy.ndarray) -> numpy.ndarray:
        """"
            :param input_img: numpy.ndarray, input image
            :return numpy.ndarray

            @brief
            Blurring input image using Gaussian blur in terms of removing
            possible noises. Size of the used kernel is 3 by 3.
        """
        try:
            img_gaussian_blur: numpy.ndarray = cv2.GaussianBlur(
                src=input_img, ksize=(3, 3), sigmaX=1)

            return img_gaussian_blur
        except cv2.error:
            logging.error("Error occurred while applying Gaussian blur")

    @staticmethod
    def __coordinate_correction(coordinate_correction_val: numpy.int32,
                                original_coordinate: numpy.int32) -> \
            numpy.int32:
        """"
            :param coordinate_correction_val: numpy.int32, correction value ;
            :param original_coordinate: numpy.int32, original coordinate value ;
            :return numpy.int32

            @brief
            Private static method in which coordinate is corrected.
        """
        if original_coordinate - coordinate_correction_val > \
                coordinate_correction_val:
            corrected_coordinate: numpy.int32 = original_coordinate - \
                                                coordinate_correction_val
        else:
            corrected_coordinate: numpy.int32 = original_coordinate + \
                                                coordinate_correction_val

        return corrected_coordinate
