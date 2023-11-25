import matplotlib.pyplot as plt
import os
from training import runTraining
from pseudo_label import fixmatch

runTraining(100)

_model_path = os.path.join("models", "ComplexUNet",
                           "_".join([str(29), "Epoch"]))

lossTotalTraining, lossTotalVal, batch_size, batch_size_val, lrs, lr = fixmatch(
    100, _model_path)


def get_Training_learning(training_loss, validation_loss, learning_rate, all_learning_rate,
                          training_batch_size, validation_batch_size, opt_name="Adam"):

    # validation_loss.pop(0)
    # training_loss.pop(0)
    files = os.listdir('./models/Test_Model')

    epochs = range(len(files))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_loss, 'r', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    info_text = (
        f"Learning Rate: {learning_rate:.5f}\n"
        f"All Learning Rates: {', '.join(map(str, all_learning_rate))}\n"
        f"Training Batch Size: {training_batch_size}\n"
        f"Validation Batch Size: {validation_batch_size}\n"
        f"Optimizer: {opt_name}"
    )

    plt.annotate(info_text, xy=(0.02, 0.92), xycoords='axes fraction', fontsize=10, bbox=dict(
        boxstyle="round,pad=0.3", edgecolor="white", facecolor="white"))

    plt.savefig("./Results/learning.png")
    plt.show()


get_Training_learning(lossTotalTraining, lossTotalVal, lr,
                      lrs, batch_size, batch_size_val, opt_name="Adam")
