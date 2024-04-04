from train_gan import train_model

train_model(64, 10, False, False, False, 100, "RGB", "basic_rgb")
train_model(64, 10, True, False, False, 100, "RGB", "patch_rgb")
train_model(64, 10, False, True, False, 100, "RGB", "unet_rgb")
