import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medicalDataLoader
from utils import *
from multi_UNet_complex import *
from UNet_Base import *

# Select GPU if available, else CPU
device = torch.device('cpu')


def loss_function(loss1, loss2, alpha: float, predictions, y_hat_1, y_hat_2):
    if (alpha > 1 or alpha < 0):
        Exception("alpha is outside [0;1]")
    _loss1 = loss1(predictions, y_hat_1)
    _loss2 = loss2(predictions, y_hat_2)
    return alpha * _loss1 + (1-alpha) * _loss2


def runTraining(epoch_num, weights_path='', augm=False):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 16
    batch_size_unlabel = 64
    batch_size_val = 16
    lr = 0.001   # Learning Rate
    epoch = epoch_num  # Number of epochs

    root_dir = './Data/'

    print(' Dataset: {} '.format(root_dir))

    # DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                           root_dir,
                                                           transform=transform,
                                                           mask_transform=mask_transform,
                                                           augment=augm,  # Set to True to enable data augmentation
                                                           equalize=False)

    train_loader_full = DataLoader(train_set_full,
                                   batch_size=batch_size,
                                   worker_init_fn=np.random.seed(0),
                                   num_workers=0,
                                   shuffle=True)

    pseudo_label_set_full = medicalDataLoader.MedicalImageDataset('unlabeled',
                                                                  root_dir,
                                                                  transform=transform,
                                                                  mask_transform=mask_transform,
                                                                  augment=False,  # Set to True to enable data augmentation
                                                                  equalize=False)

    pseudo_label_loader_full = DataLoader(pseudo_label_set_full,
                                          batch_size=batch_size_unlabel,
                                          worker_init_fn=np.random.seed(0),
                                          num_workers=0,
                                          shuffle=False)

    unlabel_set_full = medicalDataLoader.MedicalImageDataset('unlabeled',
                                                             root_dir,
                                                             transform=transform,
                                                             mask_transform=mask_transform,
                                                             augment=True,  # Set to True to enable data augmentation
                                                             equalize=False)

    unlabel_loader_full = DataLoader(unlabel_set_full,
                                     batch_size=batch_size_unlabel,
                                     worker_init_fn=np.random.seed(0),
                                     num_workers=0,
                                     shuffle=False)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=True)

    # INITIALIZE YOUR MODEL
    num_classes = 4  # NUMBER OF CLASSES

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    modelName = 'ComplexUNet'
    print(" Model Name: {}".format(modelName))

    # CREATION OF YOUR MODEL
    model = UNet(num_classes)

    # net = UNet(num_classes)
    model = model.to(device)  # Move the model to the device

    # Load the weights from the previously trained model
    if weights_path != '':
        # previous_model_dir = './models/' + 'Test_Model' + '/' + str(epoch_num) +'_Epoch'
        model.load_state_dict(torch.load(weights_path))
        print(" Model loaded: {}".format(weights_path))

    print("Total params: {0:,}".format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softMax = torch.nn.Softmax(dim=-1)
    CE_loss = torch.nn.CrossEntropyLoss()
    DiceLossV2Train = DiceLossV2(n_classes=num_classes)

    # PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        model.cuda()
        softMax.cuda()
        CE_loss.cuda()

    # DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def lambda1(epoch): return 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    ### To save statistics ####
    lossTotalTraining = []
    lossTotalVal = []
    Best_loss_val = 1000
    BestEpoch = 0
    lrs = []
    patience = 3

    directory = 'Results/Statistics/' + modelName

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if os.path.exists(directory) == False:
        os.makedirs(directory)

    # START THE TRAINING

    # FOR EACH EPOCH
    for i in range(epoch):
        model.train()
        lossEpoch = []
        lrEpoch = []
        num_batches = len(train_loader_full)

        # Supervised training
        # FOR EACH BATCH
        for j, data in enumerate(train_loader_full):
            # Set to zero all the gradients
            model.zero_grad()
            optimizer.zero_grad()

            # GET IMAGES, LABELS and IMG NAMES
            images, labels, _ = data

            # From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ################### Train ###################
            # -- The CNN makes its predictions (forward pass)
            net_predictions = model.forward(images)
            y_pred = softMax(net_predictions)
            # -- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE

            segmentation_classes = getTargetSegmentation(labels)
            seg_one_hot = F.one_hot(
                segmentation_classes, num_classes=4).permute(0, 3, 1, 2).float()

            # COMPUTE THE LOSS
            loss = loss_function(CE_loss, DiceLossV2Train,
                                 0.4, y_pred, seg_one_hot, segmentation_classes)

            lossTotal = loss
            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            loss.backward()
            optimizer.step()

            # Update LR
            # scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lrEpoch.append(lr_step)

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())

            # scheduler.step()
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, lr: {} ".format(lossTotal, lr_step))

        # Semi Supervised
        # pseudo-labeling
        model.eval()
        pseudo_labels = []
        for j, data in enumerate(pseudo_label_loader_full):
            images, _, _ = data
            pseudo_label = model(images)
            pseudo_labels.append(pseudo_label)

        model.train()
        if (len(pseudo_label) != len(unlabel_loader_full)):
            Exception(
                f"Pseudo-label size ({len(pseudo_label)}) different from Unlabed images size ({len(unlabel_loader_full)})")

        for j, data in enumerate(unlabel_loader_full):
            model.zero_grad()
            optimizer.zero_grad()
            images, _, _ = data
            labels = getTargetSegmentation(pseudo_labels[j])

            y_pred = model(images)
            s_pred = softMax(y_pred)

            loss = CE_loss(s_pred, labels)

            lossTotal = loss
            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            loss.backward()
            optimizer.step()

            # Update LR
            # scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lrEpoch.append(lr_step)

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())

            # scheduler.step()
            printProgressBar(j + 1, num_batches,
                             prefix="[Semi-Supervised] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, lr: {} ".format(lossTotal, lr_step))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossVal = inference(model, val_loader, loss_function, "modele", i)
        scheduler.step()
        lossTotalVal.append(lossVal)
        lossTotalTraining.append(lossEpoch)

        lrEpoch = np.asarray(lrEpoch)
        lrEpoch = lrEpoch.mean()
        lrs.append(lrEpoch)

        printProgressBar(num_batches, num_batches,
                         done="[Training] Epoch: {}, LossT: {:.4f}, LossV: {:.4f}".format(i, lossEpoch, lossVal))

        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

        if lossVal > Best_loss_val:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                torch.save(model.state_dict(), './models/' +
                           modelName + '/' + str(i) + '_Epoch')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0
        torch.save(model.state_dict(), './models/' +
                   modelName + '/' + str(i) + '_Epoch')
        Best_loss_val = lossVal
    return lossTotalTraining, lossTotalVal, batch_size, batch_size_val, lrs, lr
