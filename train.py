import warnings
from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

import medicalDataLoader
import argparse
from utils import *

from UNet_Base import *
import random
import torch
import pdb

import torch
print(torch.version.cuda)

# Select GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

warnings.filterwarnings("ignore")


# Loading Data
def loadData(mode, batchSize, rootDir='./Data', _augment=False, _shuffle=False, _equalize=False, _numWorkers=0):
    print("~~~~~~~~~~~ Loading Data ~~~~~~~~~~")

    root_dir = './Data/'

    print(' Dataset: {} '.format(root_dir))

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


def initModel(numberOfClasses, weights_path, modelName):
    # INITIALIZE YOUR MODEL

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    print(" Model Name: {}".format(modelName))

    # CREATION OF YOUR MODEL
    net = UNet(numberOfClasses)
    net = net.to(device)  # Move the model to the device

    # Load the weights from the previously trained model
    if weights_path != '':
        net.load_state_dict(torch.load(weights_path))
        print(" Model loaded: {}".format(weights_path))

    print("Total params: {0:,}".format(sum(p.numel()
          for p in net.parameters() if p.requires_grad)))
    return net


def gpuResources(net, softmax, CE_loss):
    # PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        net.cuda()
        softmax.cuda()
        CE_loss.cuda()


def supervise(epoch_num, weights_path='', augm=False):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 16
    batch_size_val = 16
    lr = 0.001   # Learning Rate
    epoch = epoch_num  # Number of epochs

    train_loader_full = loadData(
        'train', batch_size, _augment=augm, _shuffle=True)
    val_loader = loadData('val', batch_size_val, _augment=augm)

    modelName = 'Test_Model'
    net = initModel(4, weights_path, modelName)

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softmax = torch.nn.Softmax(dim=1)
    CE_loss = torch.nn.CrossEntropyLoss()

    gpuResources(net, softmax, CE_loss)

    # DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    ### To save statistics ####
    lossTotalTraining = []
    Best_loss_val = 1000
    BestEpoch = 0

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    # Create saving directory if not already existing
    directory = 'Results/Statistics/' + modelName
    if os.path.exists(directory) == False:
        os.makedirs(directory)

    # START THE TRAINING

    # FOR EACH EPOCH
    for i in range(epoch):
        net.train()
        lossEpoch = []
        DSCEpoch = []
        DSCEpoch_w = []
        num_batches = len(train_loader_full)

        # FOR EACH BATCH
        for j, data in enumerate(train_loader_full):
            # Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            # GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data

            # From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ################### Train ###################
            # -- The CNN makes its predictions (forward pass)
            net_predictions = net(images)

            # -- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
            segmentation_classes = getTargetSegmentation(labels)

            # COMPUTE THE LOSS
            # XXXXXX and YYYYYYY are your inputs for the CE
            CE_loss_value = CE_loss(
                softmax(net_predictions), segmentation_classes)
            lossTotal = CE_loss_value

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            lossTotal.backward()
            optimizer.step()

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, ".format(lossTotal))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)

        printProgressBar(num_batches, num_batches,
                         done="[Training] Epoch: {}, LossG: {:.4f}".format(
                             i, lossEpoch) + '\n'
                         + "[Validation] Epoch: {}, LossG: {:.4f}".format(i, lossEpoch))

        # THIS IS HOW YOU WILL SAVE THE TRAINED MODELS AFTER EACH EPOCH.
        # WARNING!!!!! YOU DON'T WANT TO SAVE IT AT EACH EPOCH, BUT ONLY WHEN THE MODEL WORKS BEST ON THE VALIDATION SET!!
        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

        torch.save(net.state_dict(), './models/' +
                   modelName + '/' + str(i) + '_Epoch')

        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)
