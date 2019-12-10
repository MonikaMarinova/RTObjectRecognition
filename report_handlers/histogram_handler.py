"""
author: Monika Marinova
version: 1.0
date: 10.12.2019
python version: 3.6
openCV version: 4.7.12
"""
import logging

import cv2
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import numpy


# pylint: disable=R0903, W0703, I1101
class HistogramHandler:
    """"
        @ brief Class in which is implemented logic for
        calculation of RGB histograms
    """

    def __init__(self, bins=16, line_width=3, alpha=0.5, resize_width=0):
        """"
             @brief Class instantiation: setup of class attributes
         """
        # private attribute for bins per channel
        self.__bins: int = bins
        # private attribute for plotted line width
        self.__line_width: int = line_width
        # private attribute for line transparency
        self.__alpha: float = alpha
        # private attribute for resizing of frame
        self.__resize_width: int = resize_width
        # private attributes for figure
        self.__figure: matplotlib.figure.Figure = None
        # private attributes for axis
        self.__axis: matplotlib.axes = None
        try:
            self.__figure, self.__axis = matplotlib.pyplot.subplots()
        except RuntimeWarning:
            logging.error("Runtime warning occurred during instantiation of "
                          "HistogramHandler object.")

    def __set_histogram_labels(self) -> None:
        """"
            :return None

            @brief Private class method in which hsistogram
            title and axis labels are set
        """
        try:
            # setting histogram name
            self.__axis.set_title("RGB Histogram")
            # setting axis labels
            self.__axis.set_xlabel('Bin')
            self.__axis.set_ylabel('Frequency')
        except Exception:
            logging.error("Error occurred during setting histogram labels.")

    def __set_axis_limits(self) -> None:
        """"
            :return None

            @brief Private class method in which axis
            limits are set
        """
        try:
            # setting x axis limit
            self.__axis.set_xlim(0, self.__bins - 1)
            # setting y axis limit
            self.__axis.set_ylim(0, 1)
        except Exception:
            logging.error("Error occurred during setting axis limits.")

    def __initialize_plot_lines(self) -> tuple:
        """"
            :return tuple of 2D Lines

            @brief Private class method in which plot
             lines are initialized
        """
        # initializing plot lines
        try:
            line_r, = self.__axis.plot(numpy.arange(self.__bins),
                                       numpy.zeros((self.__bins,)),
                                       c='r', lw=self.__line_width,
                                       alpha=self.__alpha)
            line_g, = self.__axis.plot(numpy.arange(self.__bins),
                                       numpy.zeros((self.__bins,)),
                                       c='g', lw=self.__line_width,
                                       alpha=self.__alpha)
            line_b, = self.__axis.plot(numpy.arange(self.__bins),
                                       numpy.zeros((self.__bins,)),
                                       c='b', lw=self.__line_width,
                                       alpha=self.__alpha)

            return line_r, line_g, line_b
        except Exception:
            logging.error("Error occurred during initializin of plot lines ")

    def __resize_frame_width(self, image: numpy.ndarray) -> numpy.ndarray:
        """"
            :param image: numpy.ndarray, static image - grabbed frame
            from video sequence
            :return numpy.ndarray

            @brief Private class method for resizing input frame
            for generating histograms

        """

        (height, width) = image.shape[:2]
        __resize_height: int = int(float(self.__resize_width / width)
                                   * height)
        try:
            resize_image: numpy.ndarray = cv2.resize(image,
                                                     (self.__resize_width,
                                                      __resize_height),
                                                     interpolation=cv2.cv2.
                                                     INTER_AREA)
            return resize_image
        except cv2.error:
            logging.error("Error occurred during resizing image")

    def __calculate_histograms(self, image: numpy.ndarray) -> tuple:
        """"
            :param image: numpy.ndarray, static image - grabbed frame
            from video sequence
            :return tuple

            @ brief Private class method in which histograms for the R, G, B
            channels are calculated
        """
        # get resized frame
        if self.__resize_width > 0:
            image: numpy.ndarray = self.__resize_frame_width(image)

        # get number of pixels in image
        pixels_number: int = numpy.prod(image.shape[:2])

        try:
            # unpack image channels
            (b_channel, g_channel, r_channel) = cv2.split(image)
        except cv2.error:
            logging.error("Error occurred during splitting image channels")

        try:
            # calculate histograms for r, g, b channels
            histogram_r = cv2.calcHist([r_channel], [0], None, [self.__bins],
                                       [0, 255]) / pixels_number
            histogram_g = cv2.calcHist([g_channel], [0], None, [self.__bins],
                                       [0, 255]) / pixels_number
            histogram_b = cv2.calcHist([b_channel], [0], None, [self.__bins],
                                       [0, 255]) / pixels_number

            return histogram_r, histogram_g, histogram_b
        except cv2.error:
            logging.error("Error occurred during calculating histograms")

    def __save_plot_figure(self, frame_number: int) -> None:
        """"
            :param frame_number: int, number of captured frame
            :return None

            @ brief Private class method in which file name is set
            and matplotlib.figure object is saved as png image
        """
        plot_name: str = 'results\\histograms\\frame_' + str(frame_number) \
                         + "_histogram_plot.png"
        try:
            self.__figure.savefig(plot_name)
        except IOError:
            logging.error("Error occurred during saving histogram")

    def generate_rgb_histogram(self, frame: numpy.ndarray,
                               frame_number: int) -> None:
        """"
            :param frame: numpy.ndarray, static image - grabbed frame
            from video sequence
            :param frame_number: int, number of captured frame
            :return None

            @brief Public class method in which is implemented pipe line
            for labeling, calculation and saving histogram for RGB image
        """
        try:
            # setting histogram labels
            self.__set_histogram_labels()

            # initialize histogram lines
            (red_line, green_line, blue_line) = \
                self.__initialize_plot_lines()

            # setting x and y axis limits
            self.__set_axis_limits()

            # display plot window -> functioning but slows down the video
            # HistogramHandler.__display_plot_window()

            # calculate r, g, b histograms
            (red_histogram, green_histogram, blue_histogram) = \
                self.__calculate_histograms(frame)

            # setting histogram lines data
            red_line.set_ydata(red_histogram)
            green_line.set_ydata(green_histogram)
            blue_line.set_ydata(blue_histogram)

            # draw histogram
            # self.__figure.draw

            # save histogram as png images
            self.__save_plot_figure(frame_number)
        except Exception:
            logging.error("Error occurred during generation histogram")

    @staticmethod
    def __display_plot_window() -> None:
        """"
            :return None

            @brief Private static method for visualizing
            plot window
        """
        try:
            matplotlib.pyplot.ion()
            matplotlib.pyplot.show()
        except Exception:
            logging.error("Error occurred during visualizing plot window")
