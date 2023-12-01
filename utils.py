import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
from progressBar import printProgressBar
import pdb
from os.path import isfile, join
from medpy.metric.binary import dc
from PIL import ImageOps, Image
from torchvision import transforms
import tensorflow

from random import random
# from scipy.spatial.distance import directed_hausdorff


labels = {0: 'Background', 1: 'Foreground'}


def computeDSC(pred, gt):
    dscAll = []
    pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dscAll.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    DSC = np.asarray(dscAll)

    return DSC.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(
            imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


def inference(net, img_batch, loss_function, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()
    DiceLossV2Train = DiceLossV2(n_classes=4).cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(
            i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        pred_y = softMax(net_predictions)

        segmentation_classes = getTargetSegmentation(labels)
        seg_one_hot = F.one_hot(segmentation_classes,
                                num_classes=4).permute(0, 3, 1, 2).float()

        loss = loss_function(CE_loss, DiceLossV2Train, 0.4,
                             pred_y, seg_one_hot, segmentation_classes)
        losses.append(loss.cpu().data.numpy())
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join('./Results/Images/', modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.view(
                labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


def augment_images(imgs, masks):
    _imgs = torch.empty(0)
    _masks = torch.empty(0)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    imgs = imgs.squeeze(1).data

    masks = masks.view(
        imgs.shape[0], 1, 256, 256).data / 3.0
    masks = masks.squeeze(1).data
    for (img, mask) in zip(imgs, masks):
        img = Image.fromarray(img.numpy())
        mask = Image.fromarray(mask.numpy())
        if random() > 0.5:
            mask = ImageOps.flip(mask)
            img = ImageOps.flip(img)
        if random() > 0.5:
            mask = ImageOps.mirror(mask)
            img = ImageOps.mirror(img)
        if random() > 0.5:
            angle = random() * 60 - 30
            mask = mask.rotate(angle)
            img = img.rotate(angle)
        _imgs = torch.cat((_imgs, transform(img)), dim=0)
        _masks = torch.cat((_masks, transform(mask)), dim=0)

    return _imgs.unsqueeze(1), _masks.unsqueeze(1)


def load_images(imgs_paths):
    for img_path in imgs_paths:
        img = Image.open(img_path)


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


class DiceLossV2(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossV2, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
