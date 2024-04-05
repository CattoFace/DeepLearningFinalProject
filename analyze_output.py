import matplotlib.pyplot as plt
import numpy as np

results = np.loadtxt("results/final_unet_RGB/test_results.txt", delimiter=",")
gen_disc_train = results[:, 0]
gen_disc_test = results[:, 1]
disc_train = results[:, 2]
disc_test = results[:, 3]
gen_l1_train = results[:, 4]
gen_l1_test = results[:, 5]
acc_disc = results[:, 6] * 100
fid = results[:, 7]


def plot_lines(train_data, test_data, title, x_label, file_name):
    epochs = len(test_data)
    plt.title(title)
    plt.xlabel(x_label)
    if train_data is not None:
        plt.plot(np.arange(epochs), train_data, color="red", label="Train")
    plt.plot(np.arange(epochs), test_data, color="blue", label="Test")
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.close()


plot_lines(disc_train, disc_test, "Discriminator Loss", "loss", "disc_loss.png")
plot_lines(gen_l1_train, gen_l1_test, "L1 Loss", "loss", "l1_loss.png")
plot_lines(None, fid, "FID", "FID", "FID.png")
plot_lines(None, acc_disc, "Discriminator Accuracy", "accuracy %", "accuracy.png")
