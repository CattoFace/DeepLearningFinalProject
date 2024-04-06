# Usage:
Place all the dataset images in a folder called dataset(we used https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Change any desired value in preprocess_dataset.py or preprocess_dataset_rgb.py(namely the ycbcr/yuv boolean in the former and the resize dimension).
Run preprocess_dataset.py.
Change any desired model/training parameter in train_gan.py.
Run train_gan.py.
When training is over, run demo_gan.py or demo_gan_rgb.py
