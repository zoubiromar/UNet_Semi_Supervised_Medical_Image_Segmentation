# Debug and logger helper
import warnings

# Torch imports
import torch

# Local imports
from src.UNet_Base import *
from src.utils import *
from src.progressBar import printProgressBar

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s:%(name)s]\t%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

warnings.filterwarnings("ignore")

PRINT_EVERY_N_EPOCH = 1


def supervise(epoch_num, weights_path='', augm=False):
    logger.info("Start Training")

    # DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 16
    learning_rate = 0.001
    nb_class = 4

    # Loading training data
    logger.info("Loading Data")
    X = loadData(
        'train', batch_size, _augment=augm, _shuffle=True)

    # Initiating the model
    modelName = 'Test'
    model = initModel(nb_class, weights_path, modelName)

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softmax = torch.nn.Softmax(dim=-1)
    CE_loss = torch.nn.CrossEntropyLoss()

    gpuResources(model, softmax, CE_loss)

    # DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### To save statistics ####
    lossTotalTraining = []

    # Create saving directory if not already existing
    directory = 'Results/Statistics/' + modelName
    if os.path.exists(directory) == False:
        os.makedirs(directory)

    # START THE TRAINING
    logger.info("Starting backpropagation iterations")
    # FOR EACH EPOCH
    for epoch in range(epoch_num):
        model.train()
        lossEpoch = []
        num_batches = len(X)

        # FOR EACH BATCH
        for j, data in enumerate(X):
            # Set to zero all the gradients
            model.zero_grad()
            optimizer.zero_grad()

            # GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data

            # From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ################### Train ###################
            # -- The CNN makes its predictions (forward pass)
            y_predicted = model(images)
            # -- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
            y_hat = getTargetSegmentation(labels)

            # COMPUTE THE LOSS
            # XXXXXX and YYYYYYY are your inputs for the CE
            lossTotal = CE_loss(
                softmax(y_predicted), y_hat)

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            lossTotal.backward()
            optimizer.step()

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())
            if (epoch % PRINT_EVERY_N_EPOCH == 0):
                printProgressBar(j + 1, num_batches,
                                 prefix="[Training] Epoch: {} ".format(epoch),
                                 length=15,
                                 suffix=" Loss: {:.4f}, ".format(lossTotal))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)
        if (logger.level == logging.DEBUG):
            printProgressBar(num_batches, num_batches, done="[Training] Epoch: {}, LossG: {:.4f}".format(
                epoch, lossEpoch) + '\n'
                + "[Validation] Epoch: {}, LossG: {:.4f}".format(epoch, lossEpoch))

        # THIS IS HOW YOU WILL SAVE THE TRAINED MODELS AFTER EACH EPOCH.
        # WARNING!!!!! YOU DON'T WANT TO SAVE IT AT EACH EPOCH, BUT ONLY WHEN THE MODEL WORKS BEST ON THE VALIDATION SET!!
        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

        torch.save(model.state_dict(), './models/' +
                   modelName + '/' + str(epoch) + '_Epoch')

        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)