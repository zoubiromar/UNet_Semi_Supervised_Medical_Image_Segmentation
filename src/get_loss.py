# Torch imports
import torch

# Local imports
from src.UNet_Base import *
from src.utils import *

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s:%(name)s]\t%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# Output the training and validation loss of given weights


def get_loss(weights_path):
    # Load the weights from the file
    model = initModel(4, weights_path)

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softMax = torch.nn.Softmax(dim=1)
    CE_loss = torch.nn.CrossEntropyLoss()

    model.eval()
    batch_size = 16

    validation_data = loadData('val', batch_size)
    loss = []

    for j, validation_data in enumerate(validation_data):
        # GET IMAGES, LABELS and IMG NAMES
        images, labels, img_names = validation_data

        # From numpy to torch variables
        labels = to_var(labels)
        images = to_var(images)

        ################### Validate ###################
        # -- The CNN makes its predictions (forward pass)
        y_predicted = model(images)

        # -- Compute the losses --#
        # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
        y_hat = getTargetSegmentation(labels)

        # COMPUTE THE LOSS
        # XXXXXX and YYYYYYY are your inputs for the CE
        CE_loss_value = CE_loss(softMax(y_predicted), y_hat)

        loss.append(CE_loss_value.item())

    loss = np.asarray(loss)
    loss = loss.mean()

    # print('Training loss:', lossEpoch_train)
    print('Validation loss:', loss)
    return loss
