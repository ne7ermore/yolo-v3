import os
import random

import torch
from torch.autograd import Variable
from torchvision import transforms as T

from PIL import Image, ImageDraw, ImageFont


class IMGProcess(object):
    def __init__(self, source,
                 use_cuda=True,
                 img_path="imgs",
                 batch_size=100,
                 img_size=416,
                 confidence=0.5,
                 rebuild=True,
                 result="result"):

        self.colors = source["pallete"]
        self.num_classes = source["num_classes"]
        self.classes = source["classes"]
        self.confidence = confidence
        self.rebuild = rebuild
        self.result = result
        self.use_cuda = use_cuda
        self.img_size = img_size
        self.font = ImageFont.truetype("arial.ttf", 15)
        self.imgs = [os.path.join(img_path, img)
                     for img in os.listdir(img_path)]
        self.sents_size = len(self.imgs)
        self.bsz = min(batch_size, len(self.imgs))
        self._step = 0
        self._stop_step = self.sents_size // self.bsz

    def _encode(self, x):
        encode = T.Compose([T.Resize((self.img_size, self.img_size)),
                            T.ToTensor()])

        return encode(x)

    def img2Var(self, imgs):
        self.imgs = imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs_dim = torch.FloatTensor([img.size for img in imgs]).repeat(1, 2)

        with torch.no_grad():
            tensors = [self._encode(img).unsqueeze(0) for img in imgs]
            vs = Variable(torch.cat(tensors, 0))
            if self.use_cuda:
                vs = vs.cuda()
                imgs_dim = imgs_dim.cuda()

        return vs, imgs_dim

    def predict(self, prediction, nms_conf=0.4):
        """
        prediction:
            0:3 - x, y, h, w
            4 - confidence
            5: - class score
        """

        def iou(box1, box2):
            x1, y1 = box1[:, 0], box1[:, 1]
            b1_w, b1_h = box1[:, 2] - x1 + .1, box1[:, 3] - y1 + .1

            x2, y2, = box2[:, 0], box2[:, 1]
            b2_w, b2_h = box2[:, 2] - x2 + .1, box2[:, 3] - y2 + .1

            end_x = torch.min(x1 + b1_w, x2 + b2_w)
            start_x = torch.max(x1, x2)

            end_y = torch.min(y1 + b1_h, y2 + b2_h)
            start_y = torch.max(y1, y2)

            a = (end_x - start_x) * (end_y - start_y)

            return a / (b1_w * b1_h + b2_w * b2_h - a)

        conf_mask = (prediction[:, :, 4] >
                     self.confidence).float().unsqueeze(2)
        prediction = prediction * conf_mask

        box_corner = prediction.new(*prediction.size())
        box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
        box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
        box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
        box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
        prediction[:, :, :4] = box_corner[:, :, :4]

        outputs = []

        for index in range(prediction.size(0)):
            image_pred = prediction[index]  # [10647, 85]

            max_score, max_index = torch.max(
                image_pred[:, 5:], 1, keepdim=True)
            image_pred = torch.cat(
                (image_pred[:, :5], max_score, max_index.float()), 1)  # [10647, 7]

            non_zero_ind = (torch.nonzero(image_pred[:, 4])).view(-1)

            if non_zero_ind.size(0) == 0:
                continue

            image_pred_ = image_pred[non_zero_ind, :]
            img_classes = torch.unique(image_pred_[:, -1])

            objects, img_preds = [], []
            name = self.this_img_names[index].split("/")[-1]

            for c in img_classes:
                c_mask = image_pred_ * \
                    (image_pred_[:, -1] == c).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(c_mask[:, -2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                _, conf_sort_index = torch.sort(
                    image_pred_class[:, 4], descending=True)
                image_pred_class = image_pred_class[conf_sort_index]

                for i in range(image_pred_class.size(0) - 1):
                    try:
                        ious = iou(image_pred_class[i].unsqueeze(
                            0), image_pred_class[i + 1:])
                    except IndexError:
                        break

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    non_zero_ind = torch.nonzero(
                        image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(
                        -1, 7)

                img_preds.append(image_pred_class)
                objects += [self.classes[int(x[-1])] for x in image_pred_class]

            outputs.append((name, objects))
            img_preds = torch.cat(img_preds, dim=0)

            if self.rebuild:
                self.tensor2img(img_preds, index, name)

        return outputs

    def tensor2img(self, tensor, index, name):
        imgs_dim = self.imgs_dim[index] / self.img_size
        img = self.imgs[index]
        draw = ImageDraw.Draw(img)

        tensor[:, :4] = tensor[:, :4].clamp_(0, self.img_size) * imgs_dim
        for t in tensor:
            s_x, s_y, e_x, e_y = list(map(int, t[:4]))
            label = self.classes[int(t[-1])]
            color = random.choice(self.colors)
            draw.rectangle([s_x, s_y, e_x, e_y], outline=color)
            draw.text([s_x, s_y], label, fill=color, font=self.font)

        del draw

        img.save(os.path.join(self.result, "res_{}".format(name)))

    def __iter__(self):
        return self

    def __next__(self):
        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _s = self._step * self.bsz
        self._step += 1

        self.this_img_names = self.imgs[_s:_s + self.bsz]

        vs, self.imgs_dim = self.img2Var(self.this_img_names)

        return vs
