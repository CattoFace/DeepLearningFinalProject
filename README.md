# Usage:
## Training:
Place all the dataset images in a folder called dataset(we used https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Change any desired value in preprocess_dataset.py or preprocess_dataset_rgb.py(namely the YCbCr/HSV boolean in the former and the resize dimension).
Run preprocess_dataset.py/preprocess_dataset_rgb.py
Change any desired model/training parameter in train_gan.py.
Run train_gan.py.
When training is over, run demo_gan.py or demo_gan_rgb.py

## Evaluation Only:
Place all the dataset images in a folder called dataset(we used https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Run preprocess_dataset_rgb.py.
Run eval_gan.py.
