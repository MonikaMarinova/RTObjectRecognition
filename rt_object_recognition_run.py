"""
author: Monika Marinova
version: 1.0
date:
python version: 3.6
openCV version: 4.7.12
"""
from object_recognition_processing.object_recognition_processor import \
    ObjectRecognition


def object_recognition_main() -> None:
    """"
        :return None

        @ brief
        Public method in which object with type ObjectRecognition
        is created and real time object recognition
        workflow is started.
    """
    obj_recognition = ObjectRecognition()
    ObjectRecognition.real_time_object_recognition(obj_recognition)


if __name__ == "__main__":
   object_recognition_main()
