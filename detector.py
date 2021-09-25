import argparse
from datetime import datetime
import os
import shutil
import glob
import cv2
import numpy as np
from gitdownloader import Downloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading...')
import tensorflow as tf

if not os.path.exists('object_detection'):
    print("Can't find object_detection dir\nDo you want to download it?[Y]/n")
    answer = input('>>>')
    if answer == '' or answer.lower() == 'y':
        print('Downloading...')
        downloader = Downloader('https://github.com/tensorflow/models/tree/master/research/object_detection')
        downloader.download()
        shutil.move('research/object_detection', '.')
        os.remove('research')
    else:
        exit(0)

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


class Detector:
    @classmethod
    def activate(cls):
        try:
            cls.model = tf.saved_model.load("saved_model")
            cls.category_index = label_map_util.create_category_index_from_labelmap("labelmap.pbtxt",
                                                                                    use_display_name=True)
        except FileNotFoundError:
            print("Can't find labelmap file. Ending...")
            exit(1)
        except OSError:
            print("Can't find model dir. Ending...")
            exit(1)

    def __init__(self):
        self.image = None

    def detect(self):

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = np.asarray(self.image, dtype="uint8")
        input_tensor = tf.convert_to_tensor(self.image)
        input_tensor = input_tensor[tf.newaxis, ...]

        self.output_dict = self.model(input_tensor)

        num_detections = int(self.output_dict.pop('num_detections'))
        self.output_dict = {key: value[0, :num_detections].numpy()
                            for key, value in self.output_dict.items()}
        self.output_dict['num_detections'] = num_detections

        self.output_dict['detection_classes'] = self.output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in self.output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                self.output_dict['detection_masks'], self.output_dict['detection_boxes'],
                self.image.shape[0], self.image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.7,
                                               tf.uint8)
            self.output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        vis_util.visualize_boxes_and_labels_on_image_array(
            self.image,
            self.output_dict['detection_boxes'],
            self.output_dict['detection_classes'],
            self.output_dict['detection_scores'],
            self.category_index,
            instance_masks=self.output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Source of images to analyze:\n0 - webcam\n1- images directory\n2 - video directory', default=0)
    parser.add_argument('--dir', help='Path to directory with images if source is 1. Default is images', default='images')
    parser.add_argument('--vdir', help='Path to directory with mp4 video if source is 2. Default is videos', default='videos')

    return parser.parse_args()

def run_video(directory):
    detector = Detector()
    video_name = glob.glob(f'{directory}/*.mp4')

    if len(video_name) == 1:
        video_name = video_name[0]
    else:
        print('Found more mp4 files in directory.\nType preferred filename:')
        file_name = input('>>>')
        video_name = os.path.join(directory, file_name)

    cap = cv2.VideoCapture(video_name)

    while(cap.isOpened()):
        _, frame = cap.read()
        detector.image = frame
        detector.detect()
        cv2.imshow(os.path.basename(video_name), detector.image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_images(directory):
    detector = Detector()
    files_list = os.listdir(directory)

    print("Do you want to save processed images?y/[N]")
    option = input('>>>')
    if option.lower() == 'y':
        dir_name = f'{directory}_processed'
        date = datetime.today()

        if os.path.exists(dir_name):
            dir_name = os.path.join(dir_name, date.strftime('%Y_%m_%d_%H-%M'))
        os.mkdir(dir_name)

    for file in files_list:
        image = cv2.imread(os.path.join(directory, file))
        detector.image = image
        detector.detect()
        if option.lower() == 'y':
            cv2.imwrite(os.path.join(dir_name, file.split('.')[0] + '_processed.jpg'), detector.image)
        cv2.imshow(file, detector.image)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def run_webcam():
    detector = Detector()
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        detector.image = frame
        detector.detect()
        cv2.imshow('Webcam', detector.image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    Detector.activate()

    args = get_args()

    if args.source == '0':
        run_webcam()
    elif args.source == '1':
        run_images(args.dir)
    elif args.source == '2':
        run_video(args.vdir)


if __name__ == '__main__':
    main()
