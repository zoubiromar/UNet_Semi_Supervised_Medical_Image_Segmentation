from src.supervise_model import *
from src.get_loss import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s:%(name)s]\t%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# Script begining
num_epoch = 100

logger.info(f"Train supervise model with {num_epoch} epochs")
supervise(num_epoch)


models_paths = ['./models/' + 'Test' + '/' +
                str(i) + '_Epoch' for i in range(90, 100)]
# models_paths = ['./models/' + 'Trained/Data_Augmented_15K_EpochPlus' +
#                 '/Adam_Labeled_' + str(i) + '5k_Epoch' for i in range(0, 8)]
image_path = "./Data/val/Img/patient061_01_1.png"

for model_path in models_paths:
    logger.debug(f"model's path is: {model_path}")
    print(f"Loss : {get_loss(model_path)}")
    # plotOutput(model_path, image_path)
