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


def fixmatch(epoch_num, weights_path='', augm=False):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 1
    batch_size_unlabel = 4
    batch_size_val = 4
    lr = 0.001   # Learning Rate

    root_dir = './Data_/'

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
                                                           augment=False,  # Set to True to enable data augmentation
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
                                          shuffle=True)

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
                                     shuffle=True)

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
    modelName = 'FixMatch'
    print(" Model Name: {}".format(modelName))

    # CREATION OF YOUR MODEL
    student = UNet(num_classes)
    teacher = ComplexUNet(num_classes)

    # net = UNet(num_classes)
    student = student.to(device)  # Move the model to the device
    teacher = teacher.to(device)  # Move the model to the device

    # Load the weights from the previously trained model
    if weights_path != '':
        # previous_model_dir = './models/' + 'Test_Model' + '/' + str(epoch_num) +'_Epoch'
        teacher.load_state_dict(torch.load(weights_path))
        print(" Model loaded: {}".format(weights_path))

    print("Total params: {0:,}".format(sum(p.numel()
          for p in student.parameters() if p.requires_grad)))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    soft_max = torch.nn.Softmax(dim=-1)
    ce_loss_train = torch.nn.CrossEntropyLoss()
    ce_loss_unlabeled = torch.nn.CrossEntropyLoss(label_smoothing=0.95)

    # PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        student.cuda()
        teacher.cuda()
        soft_max.cuda()
        ce_loss_train.cuda()
        ce_loss_unlabeled.cuda()

    # DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    def lambda1(epoch_num): return 0.95 ** epoch_num
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
    for epoch in range(epoch_num):
        student.train()
        lossEpoch = []
        lrEpoch = []
        num_batches = len(train_loader_full)

        # Semi Supervised
        for j, (data_train, data_pseudo_label, data_unlabeled) in enumerate(zip(train_loader_full, pseudo_label_loader_full, unlabel_loader_full)):
            student.train()
            student.zero_grad()
            optimizer.zero_grad()
            images_train, _, _ = data_train
            images_pseudo_label, _, _ = data_pseudo_label
            images_unlabeled, _, _ = data_unlabeled

            y_pred = student(images_train)
            s_pred = soft_max(y_pred)

            # create labels on which we want to match on
            with torch.no_grad():
                labels = teacher(images_train)
            s_labels = soft_max(labels)

            # COMPUTE THE LOSS
            loss_train = ce_loss_train(s_pred, s_labels)
            student.eval()
            with torch.no_grad():
                pseudo_labels = student(images_pseudo_label)
            s_labels = soft_max(pseudo_labels)
            # -- The CNN makes its predictions (forward pass)
            student.train()

            y_pred = student(images_unlabeled)
            s_pred = soft_max(y_pred)

            loss_unlabel = ce_loss_unlabeled(s_pred, s_labels)

            loss = loss_train + loss_unlabel

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            loss.backward()
            optimizer.step()

            # Update LR
            # scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lrEpoch.append(lr_step)

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(loss.cpu().data.numpy())

            # scheduler.step()
            printProgressBar(j + 1, num_batches,
                             prefix="[Semi-Supervised] Epoch: {} ".format(
                                 epoch),
                             length=15,
                             suffix=" Loss: {:.4f}, lr: {} ".format(loss, lr_step))

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossVal = inference(student, val_loader,
                            loss_function, "modele", epoch)
        scheduler.step()
        lossTotalVal.append(lossVal)
        lossTotalTraining.append(lossEpoch)

        lrEpoch = np.asarray(lrEpoch)
        lrEpoch = lrEpoch.mean()
        lrs.append(lrEpoch)

        printProgressBar(num_batches, num_batches,
                         done="[Training] Epoch: {}, LossT: {:.4f}, LossV: {:.4f}".format(epoch, lossEpoch, lossVal))

        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

        if lossVal > Best_loss_val:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                torch.save(student.state_dict(), './models/' +
                           modelName + '/' + str(epoch) + '_Epoch')
                break
        else:
            print('trigger times: 0')
            trigger_times = 0
        torch.save(student.state_dict(), './models/' +
                   modelName + '/' + str(epoch) + '_Epoch')
        Best_loss_val = lossVal
    return lossTotalTraining, lossTotalVal, batch_size, batch_size_val, lrs, lr
