import os
import shutil

import cv2
import numpy as np
from loguru import logger
import xml.etree.ElementTree as ET
import pickle
import os
from collections import OrderedDict
from os import listdir, getcwd
from os.path import join


class YOLO:
    def __init__(self, dataset_root: str = None):
        self.dataset_root = dataset_root
        self.images = os.path.join(self.dataset_root, 'images')
        self.labels = os.path.join(self.dataset_root, 'labels')
        self.train_txt = os.path.join(self.dataset_root, 'train.txt')
        self.classes_txt = os.path.join(self.dataset_root, 'classes.txt')
        assert os.path.exists(self.images), logger.error(f'images dir is not found.')
        assert os.path.exists(self.labels), logger.error(f'labels dir is not found.')
        assert os.path.exists(self.classes_txt), logger.error(f'classes.txt dir is not found.')
        assert os.path.exists(self.train_txt), logger.error(f'train.txt dir is not found.')

    def get_classes(self) -> list:
        with open(self.classes_txt, 'r') as f:
            classes = f.readlines()
        classes = [i.strip('\n') for i in classes]
        return classes

    def get_train_file(self):
        with open(self.classes_txt, 'r') as f:
            train = f.readlines()
        train = [i.strip('\n') for i in train]
        return train

    def check_image(self):
        classes = self.get_classes()
        images = os.listdir(self.images)
        for image in images:
            image_path = os.path.join(self.images, image)
            label_path = os.path.join(self.labels, image.replace('jpg', 'txt'))
            logger.info(f'image: {image_path}')
            logger.info(f'label: {label_path}')
            im = cv2.imread(image_path)
            h, w, c = im.shape

            boxes = self.parse_label(label_path, (w, h))
            for box in boxes:
                print(box)
                label = classes[int(box[-1])]
                box = box[:4]
                self.draw(im, box, label)

            cv2.imshow('image', im)
            cv2.waitKey(0)

    def parse_label(self, txt: str, wh: tuple) -> list:
        label = os.path.join(self.labels, txt)
        w, h = wh

        with open(label, 'r') as f:
            data = f.readlines()
        boxes = []
        for box in data:
            box = box.split()

            classes = int(box[0])
            x_, y_, w_, h_ = float(box[1]), float(box[2]), float(box[3]), float(box[4])

            x1 = w * x_ - 0.5 * w * w_
            x2 = w * x_ + 0.5 * w * w_
            y1 = h * y_ - 0.5 * h * h_
            y2 = h * y_ + 0.5 * h * h_

            box = [x1, y1, x2, y2, classes]
            boxes.append(box)
        return boxes

    def draw(self,
             image: np.ndarray = None,
             box: list = None,
             label: str = '',
             color: tuple = (128, 128, 128),
             txt_color: tuple = (0, 0, 0)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)


class VOC:
    def __init__(self, dataset_root: str = None, classes: list = None):
        self.classes = classes if classes is not None else []

        self.dataset_root = dataset_root
        self.image_dir = os.path.join(self.dataset_root, 'JPEGImages')
        self.annotations = os.path.join(self.dataset_root, 'Annotations')
        self.train_file = os.path.join(self.dataset_root, 'ImageSets/Main/train.txt')

        logger.info(f'dataset root: {self.dataset_root}')
        assert os.path.exists(self.image_dir), logger.error(f'image dir: {self.image_dir} is not found.')

        # xxx/labels: save 1.txt 2.txt
        self.yolo_label_dir = os.path.join(self.dataset_root, 'labels')

    def create_train_file(self):
        images = os.listdir(self.image_dir)
        print(images)
        with open(self.train_file, 'w') as f:
            for image in images:
                img_name, subfix = image.split('.')
                f.write(img_name + '\n')

    def image_check(self):
        with open(self.train_file, 'r') as f:
            train_txt = f.readlines()

        train_txt = [i.strip('\n') for i in train_txt]
        print(train_txt)
        for item in train_txt:
            image = os.path.join(self.image_dir, item + '.jpg')
            xml = image.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')

            logger.info(f'image path: {image}')
            logger.info(f'label xml path: {xml}')
            xml = self._parse_xml(xml)
            print(xml)
            image = cv2.imread(image)

            for box in xml['boxes']:
                self.draw(image, box['box'], box['name'])

            cv2.imshow('image', image)
            cv2.waitKey(0)

    def draw(self,
             image: np.ndarray = None,
             box: list = None,
             label: str = '',
             color: tuple = (128, 128, 128),
             txt_color: tuple = (0, 0, 0)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    # 解析xml文件
    def _parse_xml(self, xml: str) -> dict:
        """
        :param xml: xxx.xml
        :return:{size:,boxes:[box{name,diff,box:[]}]}
        """
        root = ET.parse(xml).getroot()

        size = root.find('size')
        data = {
            'size': [int(size.find('width').text),
                     int(size.find('height').text),
                     int(size.find('depth').text)],
        }

        boxes = []
        for obj in root.iter('object'):
            bndbox = obj.find("bndbox")

            name = str(obj.find("name").text.lower().strip())
            difficult = int(obj.find("difficult").text.lower().strip())
            box = [
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            ]
            boxes.append({'name': name, 'difficult': difficult, 'box': box})
        data['boxes'] = boxes
        return data

    # 归一化voc数据为yolo格式
    def convert(self, size: tuple, box: list):
        # size:[image_w,image_h]
        # box:[x1,y1,x2,y2]
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(self, xml: str):
        # 1.xml->1.txt
        out_file = os.path.join(self.yolo_label_dir, xml.replace('xml', 'txt'))
        logger.info(f'out file: {out_file}')

        # 解析xml
        xml = os.path.join(self.annotations, xml)
        xml_data = self._parse_xml(xml)

        size = xml_data['size']
        w = size[0]
        h = size[1]

        for obj in xml_data['boxes']:
            if obj['difficult'] == 1:
                continue
            if obj['name'] not in self.classes:
                continue

            cls_id = self.classes.index(obj['name'])

            box = obj['box']
            # bb=(x,y,w,h) 归一化数据
            bb = self.convert((w, h), box)

            with open(out_file, 'a+') as f:
                f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def voc2yolo(self):
        if os.path.exists(self.yolo_label_dir) is False:
            os.mkdir(self.yolo_label_dir)
            assert os.path.exists(self.yolo_label_dir), logger.error(f'label dir create failed.')
            logger.success(f'create {self.yolo_label_dir}')
        else:
            shutil.rmtree(self.yolo_label_dir)
            logger.success(f'remove old file {self.yolo_label_dir}')
            os.mkdir(self.yolo_label_dir)
            logger.success(f'create {self.yolo_label_dir}')
        assert os.path.exists(self.annotations), logger.error(f'Annotations is not found.')

        xmls = os.listdir(self.annotations)
        print(xmls)
        for xml in xmls:
            self.convert_annotation(xml)


if __name__ == '__main__':
    dataset_root = '/home/ubuntu/data/VOC_Fire_Smoke/VOC2020'
    classes = ['fire']
   
    voc = VOC(dataset_root, classes)
    yolo = YOLO(dataset_root)

    # voc.voc2yolo()
    yolo.check_image()
