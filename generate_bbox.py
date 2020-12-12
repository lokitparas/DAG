# -*-coding:utf-8-*-
from __future__ import division

import os
import xml.etree.ElementTree as ET

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
# from wider import WIDER
from skimage import transform as sktsf
from torch.autograd import Variable
import time

import attacks
from data.dataset import inverse_normalize
from data.dataset import pytorch_normalze
from data.util import read_image
from model import FasterRCNNVGG16
# from trainer import BRFasterRcnnTrainer
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils import make_one_hot
from attack import DAG

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
n_classes = 21
num_iterations = 20
gamma = 0.5

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
# data_dir = '/home/xingxing/liangsiyuan/data/VOCdevkit/VOC2007'
data_dir = '/u/l/p/lparas/Desktop/GAN/img_attack_with_attention/data/VOCdevkit/VOC2007'
# data_dir = '/home/xingxing/liangsiyuan/data/video_dataset'
# attacker_path = '/home/xlsy/Documents/CVPR19/final results/weights/attack_12211147_2500.path'
attacker_path = 'checkpoints/10.path'
save_path_HOME = '/u/l/p/lparas/Desktop/GAN/img_attack_with_attention/results/ssd_attack'


# save_path_HOME = '/home/xingxing/liangsiyuan/results/ssd_attack_video'


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='test',
                 use_difficult=False, return_difficult=False,
                 ):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        # id_list_file = os.listdir(data_dir)
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.save_dir = save_path_HOME + '/frcnn/'
        self.save_dir_adv = save_path_HOME + '/JPEGImages/'
        self.save_dir_perturb = save_path_HOME + '/frcnn_perturb/'

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, min_size=300, max_size=448):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.
             (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        print(img.shape)
        C, H, W = img.shape
        max_size = 400
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        try:
            # img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
            img = sktsf.resize(img, (C, 300, 300), mode='reflect')
        except:
            ipdb.set_trace()
        # both the longer and shorter should be less than
        # max_size and min_size
        normalize = pytorch_normalze
        print(img.shape)
        return normalize(img)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        # print('id of img is:' + id_)
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            # if not self.use_difficult and int(obj.find('difficult').text) == 1:
            #     continue

            # difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(np.zeros(label.shape), dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        img = self.preprocess(img)
        img = torch.from_numpy(img)[None]
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult, id_

    __getitem__ = get_example


def add_bbox(ax, bbox, label, score):
    # print(bbox)
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def img2jpg(img, img_suffix, quality):
    jpg_base = 'frcnn_adv_jpg/'
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'))
    print(img)
    if not os.path.exists(jpg_base):
        os.makedirs(jpg_base)
    jpg_path = jpg_base + img_suffix
    img.save(jpg_path, format='JPEG', subsampling=0, quality=quality)
    jpg_img = read_image(jpg_path)
    return jpg_img



def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        names.append(filename)
        img = read_image(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images, names

if __name__ == '__main__':
    layer_idx = 20
    _data = VOCBboxDataset(data_dir)
    print("data loaaded")
    faster_rcnn = FasterRCNNVGG16()
    # faster_rcnn.eval()
    faster_rcnn = faster_rcnn.cuda()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # attacker = attacks.Blade_runner()
    # attacker.load(attacker_path)
    # attacker.eval()
    print("Trainer train the model")
    device = torch.device("cuda:0")
    # trainer = BRFasterRcnnTrainer(faster_rcnn, attacker, \
    #                               layer_idx=layer_idx, attack_mode=True)
    # trainer = trainer.to(device)
    # trainer.load('/home/xlsy/Documents/CVPR19/final results/weights/fasterrcnn_img_0.701.pth')
    trainer.load('fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth')
    trainer = trainer.to(device)
    print("Something....... * * * * ** *")
    quality_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    threshold = [0.7]
    adv_det_list = []


    origImgs = load_images_from_folder('frcnn_adv_jpg/original/jpeg')
    succAdvImgs = load_images_from_folder('frcnn_adv_jpg/failed')
    failAdvImgs = load_images_from_folder('frcnn_adv_jpg/success')

    for quality in threshold:
        img_detected_total = 0
        adv_detected_total = 0
        img_object_total = 0
        total_distance = 0
        total_time = 0
        trainer.faster_rcnn.score_thresh = quality
        for i in range(len(origImgs[0])):
            # if int(img_id.split('.')[0]) > 231:
            #     continue
            # img_labels_lists = np.unique(img_labels)
            # with open('frcnn_adv_jpg/original/origLabels_' + str(i), 'wb') as q:
            #     pickle.dump(img_labels_lists, q)
            #     q.close()

            pureimg = origImgs[0][i]
            # print(img.shape)
            img = _data.preprocess(pureimg)
            ori_img_ = torch.from_numpy(img)[None]
            ori_img_ = Variable(ori_img_.float().cuda())
            print(ori_img_.shape)
            unprocessedImg = inverse_normalize(at.tonumpy(ori_img_[0]))
            unprocessedImg = torch.tensor(unprocessedImg).unsqueeze(0)
            # print(unprocessedImg.shape)
            # img = Variable(img.float().cuda())
            # ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            bboxes, labels, scores = faster_rcnn.predict(unprocessedImg, visualize=True)
            print(labels)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            final = pureimg.transpose(1,2,0).astype(int)
            # print(final.shape)
            # print(np.min(final))
            ax.imshow(final)
            # ax = add_bbox(ax,bboxes[0],labels[0],None)
            plt.savefig('bbox_imgs/original/'+origImgs[1][i])
            plt.close(fig)
            continue
            # img2jpg(inverse_normalize(attackOutput[0].cpu().data.numpy()[0]), prefix + '/adv_img_' + str(i) + '.jpg',
            #         100)

            # img_labels = torch.ones([1,1,1,1],device=device)
            # print(img_labels)
            # img_labels = torch.tensor(np.array([[img_labels[0]]]), dtype=torch.int64)
            # print(img_labels)
            # img_labels = img_labels.reshape(1,1,1)
            # print(img_labels.size(2))
            # img_labels = img_labels.to(device)
            # label_oh = make_one_hot(img_labels, n_classes, device)
            # print("one hot labels")
            # print(label_oh)
            # im_path_clone = b = '%s' % im_path
            # print(type(faster_rcnn))
            roi_cls_locs, roi_scores, rois, roi_indices = faster_rcnn(img)
            rois_num = len(rois)
            if rois_num < 300:
                print(i)
            im_path_clone = str(img_id)
            # save_path = _data.save_dir + 'f0rcnn' + im_path.split('/')[-1]
            save_path = _data.save_dir + im_path_clone.split('/')[-1] + '.jpg'
            save_path_adv = _data.save_dir_adv + im_path_clone.split('/')[-1] + '.jpg'
            save_path_perturb = _data.save_dir_perturb + 'frcnn_perturb_' + im_path_clone.split('/')[-1] + '.jpg'

            # print("Directory: " + _data.save_dir)
            if not os.path.exists(_data.save_dir):
                os.makedirs(_data.save_dir)
            if not os.path.exists(_data.save_dir_adv):
                os.makedirs(_data.save_dir_adv)
            if not os.path.exists(_data.save_dir_perturb):
                os.makedirs(_data.save_dir_perturb)

            ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            # _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_],\
            #         new_score=quality, visualize=True)
            # _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)

            # ori_img_ = torch.tensor(img).unsqueeze(0)
            ori_img_ = Variable(img.cuda(), requires_grad=True)

            adv_target = torch.zeros_like(label_oh, requires_grad=True)
            adv_target = Variable(adv_target.cuda(), requires_grad=True)

            adversarial_examples = []
            # print(ori_img_.size())

            attackOutput = DAG(model=faster_rcnn,
                                                 image=ori_img_,
                                                 ground_truth=label_oh,
                                                 adv_target=adv_target,
                                                 num_iterations=500,
                                                 gamma=gamma,
                                                 no_background=False,
                                                 background_class=0,
                                                 device=device,
                                                 verbose=True, qual=quality, actualLab=img_labels[0])
            prefix = ''
            if attackOutput[6]:
                prefix = 'failed'
            else:
                prefix = 'success'

            # with open('attackOutputs_3000/'+prefix+'/attackOutput_'+str(i), 'wb') as f:
            #     pickle.dump(attackOutput, f)
            #     f.close()
            #
            # with open('attackOutputs_3000/'+prefix+'/attackOutput_'+str(i), 'rb') as f:
            #     attackOutput = pickle.load(f)
            #     f.close()

            img2jpg(inverse_normalize(attackOutput[0].cpu().data.numpy()[0]), prefix+'/adv_img_'+str(i)+'.jpg', 100)

            print('________________________________________\n\n\n\n________________________________________')
            # before = time.time()
            # adv_img, perturb, distance = trainer.attacker.perturb(img, save_perturb=save_path_perturb, rois=rois, roi_scores=roi_scores)
            # after = time.time()
            # generate_time = after-before
            # total_time = total_time + generate_time
            # total_distance = total_distance + distance
            # adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
            # perturb_ = inverse_normalize(at.tonumpy(perturb[0]))
            # del adv_img, perturb, img
            # perturb_ = perturb_.transpose((1, 2, 0))

            # 将图片从BGR转换为RGB格式进行保存
            # perturb_RGB = cv2.cvtColor(perturb_, cv2.COLOR_BGR2RGB)
            # ori_img_RGB = cv2.cvtColor(ori_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            # adv_img_RGB = cv2.cvtColor(adv_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)

            # cv2.imwrite(save_path_perturb, perturb_RGB)
            # cv2.imwrite(save_path, ori_img_RGB)
            # cv2.imwrite(save_path_adv, adv_img_RGB)
            # print('The mean distance between ori and adv is %f' % (total_distance / i))

            #
            # adv_bboxes, adv_labels, adv_scores = trainer.faster_rcnn.predict([adv_img_], \
            #         new_score=quality, visualize=True, adv=True)

            print('generate adv for ', img_id)
        print('generate adv for all imgs')

