from train_gan import train_model

train_model(64, 10, False, False, False, 100, "HSV", "basic_hsv")
train_model(64, 10, True, False, False, 100, "HSV", "patch_hsv")
train_model(64, 10, False, True, False, 100, "HSV", "unet_hsv")
