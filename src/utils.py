import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
from src.progressBar import printProgressBar
import pdb
from os.path import isfile, join
from medpy.metric.binary import dc
from torch.utils.data import DataLoader
from torchvision import transforms
import src.medicalDataLoader as medicalDataLoader
from src.UNet_Base import *
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s:%(name)s]\t%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


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


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(
            i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
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


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


###############################################
# Initialisation developpement
logger.info(torch.version.cuda)
# Select GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.debug(device)

# Computer speed


def gpuResources(net, softmax, CE_loss):
    # PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        net.cuda()
        softmax.cuda()
        CE_loss.cuda()

###############################################
# Loading Data


def loadData(mode, batchSize, rootDir='./Data', _augment=False, _shuffle=False, _equalize=False, _numWorkers=0):
    # DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION

    _transform = transforms.Compose([
        transforms.ToTensor()
    ])

    _mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    _set = medicalDataLoader.MedicalImageDataset(mode,
                                                 rootDir,
                                                 transform=_transform,
                                                 mask_transform=_mask_transform,
                                                 augment=_augment,  # Set to True to enable data augmentation
                                                 equalize=_equalize)

    loader = DataLoader(_set,
                        batch_size=batchSize,
                        worker_init_fn=np.random.seed(0),
                        num_workers=_numWorkers,
                        shuffle=_shuffle)

    return loader


def initModel(numberOfClasses, weights_path, modelName="Test"):
    # INITIALIZE YOUR MODEL
    logger.info(f" Model Name: {modelName}")

    # CREATION OF YOUR MODEL
    net = UNet(numberOfClasses)
    net = net.to(device)  # Move the model to the device

    # Load the weights from the previously trained model
    if weights_path != '':
        net.load_state_dict(torch.load(weights_path))
        logger.info(f" Model loaded: {weights_path}")

    logger.debug("Total params: {0:,}".format(sum(p.numel()
                                                  for p in net.parameters() if p.requires_grad)))
    return net

###############################################
# Output


def plotOutput(model_path, image_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CREATION OF YOUR MODEL
    net_test = initModel(4, model_path, 'Test')

    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Preprocess the image
    x = TF.to_tensor(image)
    x = TF.normalize(x, [0.5], [0.5])
    x = x.unsqueeze(0)  # Add batch dimension

    # Move the tensor to the same device as the model
    x = x.to(device)

    output = net_test(x)

    # The output is the predicted segmentation
    predicted_segmentation = torch.argmax(output.squeeze(), dim=0)

    # Convert the tensor to a numpy array
    predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

    # Display the predicted segmentation
    plt.imshow(predicted_segmentation, cmap='gray')
    plt.show()

# Displaying the outputs for different models

###############################################
