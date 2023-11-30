import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medicalDataLoader
from utils import *
from multi_UNet_complex import *
from UNet_Base import *

# Select GPU if available, else CPU
device = torch.device('cpu')

# Same loss function that training, it is only use for validation


def loss_function(loss1, loss2, alpha: float, predictions, y_hat_1, y_hat_2):
    if (alpha > 1 or alpha < 0):
        Exception("alpha is outside [0;1]")
    _loss1 = loss1(predictions, y_hat_1)
    _loss2 = loss2(predictions, y_hat_2)
    return alpha * _loss1 + (1-alpha) * _loss2


# methode base on fixmatch article
def fixmatch(epoch_num, weights_path='', augm=False):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)
    NUM_CLASSES = 4  # NUMBER OF CLASSES
    MODEL_NAME = 'FixMatch'

    # DEFINE HYPERPARAMETERS (batch_size > 1)
    BATCH_SIZE_TRAIN = 2
    BATCH_SIZE_UNLABEL = 2
    BATCH_SIZE_VAL = 2
    ROOT_DIR = './Data_/'
    PATIENCE = 3  # PATIENCE POUR LA VALIDATION
    THRESHOLD = 0.95

    lr = 0.03   # Learning Rate
    # use a modify root directory where labeled images had been add to to unlabeled data

    print(' Dataset: {} '.format(ROOT_DIR))

    # Create directory to save models
    directory = 'Results/Statistics/' + MODEL_NAME
    if os.path.exists(directory) == False:
        os.makedirs(directory)

    # DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                           ROOT_DIR,
                                                           transform=transform,
                                                           mask_transform=mask_transform,
                                                           augment=False,  # Set to True to enable data augmentation
                                                           equalize=False)

    train_loader_full = DataLoader(train_set_full,
                                   batch_size=BATCH_SIZE_TRAIN,
                                   worker_init_fn=np.random.seed(),
                                   num_workers=0,
                                   shuffle=True)

    pseudo_label_set_full = medicalDataLoader.MedicalImageDataset('unlabeled',
                                                                  ROOT_DIR,
                                                                  transform=transform,
                                                                  mask_transform=mask_transform,
                                                                  augment=False,  # Set to True to enable data augmentation
                                                                  equalize=False)

    pseudo_label_loader_full = DataLoader(pseudo_label_set_full,
                                          batch_size=BATCH_SIZE_UNLABEL,
                                          worker_init_fn=np.random.seed(0),
                                          num_workers=0,
                                          shuffle=False)

    unlabel_set_full = medicalDataLoader.MedicalImageDataset('unlabeled',
                                                             ROOT_DIR,
                                                             transform=transform,
                                                             mask_transform=mask_transform,
                                                             augment=True,  # Set to True to enable data augmentation
                                                             equalize=False)

    unlabel_loader_full = DataLoader(unlabel_set_full,
                                     batch_size=BATCH_SIZE_UNLABEL,
                                     worker_init_fn=np.random.seed(0),
                                     num_workers=0,
                                     shuffle=False)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    ROOT_DIR,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    augment=True,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=BATCH_SIZE_VAL,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=True)

    # INITIALIZE YOUR MODEL

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    print(" Model Name: {}".format(MODEL_NAME))

    # CREATION OF YOUR MODEL
    model = ComplexUNet(NUM_CLASSES)

    model = model.to(device)  # Move the model to the device

    # Load the weights from the previously trained model
    if weights_path != '':
        model.load_state_dict(torch.load(weights_path))
        print(" Model loaded: {}".format(weights_path))

    # Logging parameters informations
    print("Total params: {0:,}".format(sum(p.numel()
          for p in model.parameters() if p.requires_grad)))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    # apply softmax on 1st dimension because it correspond to the four different classes
    soft_max = torch.nn.Softmax(dim=1)
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss_V2_train = DiceLossV2(n_classes=4)

    # PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        model.cuda()
        soft_max.cuda()
        dice_loss_V2_train.cuda()
        ce_loss.cuda()

    # DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def lambda1(epoch_num): return 0.95 ** epoch_num
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    ### To save statistics ####
    lossTotalTraining = []
    lossTotalVal = []
    best_loss_val = float('inf')
    best_epoch = 0
    lrs = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    # START THE TRAINING
    num_batches_train = len(train_loader_full)

    # FOR EACH EPOCH
    for epoch in range(epoch_num):
        lossEpoch = []
        lrEpoch = []

        # Semi Supervised
        for nth_batch_train, data_train in enumerate(train_loader_full):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            images_train, labels, _ = data_train

            images_train = to_var(images_train)
            labels = to_var(labels)

            # create labels on which we want to match on (same as trained model)
            segmentation_classes = getTargetSegmentation(labels)
            seg_one_hot = F.one_hot(
                segmentation_classes, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
            # s_labels = soft_max(labels)

            # the student prediction
            y_pred = model(images_train)

            s_pred = soft_max(y_pred)

            # COMPUTE THE LOSS for trained image
            # loss_train = ce_loss_train(s_pred, s_labels)
            loss_train = loss_function(ce_loss, dice_loss_V2_train,
                                       0.4, s_pred, seg_one_hot, segmentation_classes)

            # start concistency (compare no transformed data to transformed data using the same model)
            loss_unlabel = 0
            s_labels = None
            s_pred = None
            model.zero_grad()
            optimizer.zero_grad()
            # delete train_loader_full from iterration (added for testing for memory issues)
            for (data_pseudo_label, data_unlabeled, _) in zip(pseudo_label_loader_full, unlabel_loader_full, train_loader_full):
                images_pseudo_label, _, _ = data_pseudo_label
                images_unlabeled, _, _ = data_unlabeled
                model.eval()

                with torch.no_grad():
                    pseudo_labels = model(images_pseudo_label)
                s_label = soft_max(pseudo_labels)
                # -- The CNN makes its predictions (forward pass)
                _pourcentage_per_classes = torch.mean(
                    torch.mean(s_label, dim=-1), dim=-1)
                # if the model find a image that have more than threshold accuracy on a class add it to training set
                if (torch.min(torch.max(_pourcentage_per_classes, dim=-1).values) > THRESHOLD):
                    model.train()

                    try:
                        s_labels = torch.cat((s_labels, s_label))
                    except:
                        s_labels = s_label
                    y_pred = model(images_unlabeled)
                    try:
                        s_pred = torch.cat((s_pred, soft_max(y_pred)))
                    except:
                        s_pred = soft_max(y_pred)
            # calculate loss
            if s_pred != None and s_labels != None:
                loss_unlabel = ce_loss(s_pred, s_labels)
            else:
                loss_unlabel = 0
            # make a mix of losses
            loss = loss_train + loss_unlabel

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            loss.backward()
            optimizer.step()

            # Update LR
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lrEpoch.append(lr_step)

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(loss.cpu().data.numpy())

            printProgressBar(nth_batch_train + 1, num_batches_train,
                             prefix="[Semi-Supervised] Epoch: {} ".format(
                                 epoch),
                             length=15,
                             suffix=" Loss: {:.4f}, lr: {} ".format(loss, lr_step))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossVal = inference(model, val_loader,
                            loss_function, "modele", epoch)
        scheduler.step()
        lossTotalVal.append(lossVal)
        lossTotalTraining.append(lossEpoch)

        lrEpoch = np.asarray(lrEpoch)
        lrEpoch = lrEpoch.mean()
        lrs.append(lrEpoch)

        printProgressBar(num_batches_train, num_batches_train,
                         done="[Training] Epoch: {}, LossT: {:.4f}, LossV: {:.4f}".format(epoch, lossEpoch, lossVal))

        if not os.path.exists('./models/' + MODEL_NAME):
            os.makedirs('./models/' + MODEL_NAME)

        if lossVal > best_loss_val:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= PATIENCE:
                print('Early stopping!\nStart to test process.')
                torch.save(model.state_dict(), './models/' +
                           MODEL_NAME + '/' + str(epoch) + '_Epoch')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0
            best_epoch = epoch
        torch.save(model.state_dict(), './models/' +
                   MODEL_NAME + '/' + str(epoch) + '_Epoch')
        best_loss_val = lossVal
    return lossTotalTraining, lossTotalVal, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, lrs, lr, best_epoch
