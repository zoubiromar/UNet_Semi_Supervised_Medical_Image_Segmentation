from supervise_model import *
from get_loss import *


num_epoch = 100

supervise(num_epoch)


models_paths = ['./models/' + 'Test' + '/' +
                str(i) + '_Epoch' for i in range(90, 100)]
# models_paths = ['./models/' + 'Trained/Data_Augmented_15K_EpochPlus' +
#                 '/Adam_Labeled_' + str(i) + '5k_Epoch' for i in range(0, 8)]
image_path = "./Data/val/Img/patient061_01_1.png"

for model_path in models_paths:
    print(f"Loss : {get_loss(model_path)}")
    # plotOutput(model_path, image_path)
