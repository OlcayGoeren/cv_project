
# from torch.optim.optimizer import Optimizer, required
# import math
# import segmentation_models_pytorch as smp
# from torch.utils.data import random_split
# import os
# from torch.utils.data import Dataset
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import segmentation_models_pytorch as smp
# from torch import nn
# from torch.optim import AdamW
# from tqdm import tqdm
# from skimage import io, transform
# import cv2
# import albumentations as album
# from torch.utils.data import DataLoader

# https://www.kaggle.com/vad13irt/fish-semantic-segmentation-resnet34-unet testmest

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensor
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import logging
import numpy as np
import wandb
from addict import Dict
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


warnings.simplefilter("ignore")
logging.basicConfig(format="[%(asctime)s][%(levelname)s] - %(message)s")


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536,
                          always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class BodyDataset(Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            x_dataset,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.x_dataset = x_dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(os.path.join(self.images_dir, self.x_dataset[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, self.x_dataset[i]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        # mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        return image, mask

    def __len__(self):
        # return length of
        return len(self.x_dataset)


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":

    DATA_DIR = './bodies/'
    IMAGES_DIR = './bodies/images/'
    MASKS_DIR = './bodies/masks/'
    full_dataset = next(os.walk(
        './bodies/images'))[2]
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        full_dataset, [train_size, test_size])
    valid_dataset = np.array_split(valid_dataset, 2)[0]
    test_dataset = np.array_split(valid_dataset, 2)[1]

    select_classes = ['background', 'body']
    class_rgb_values = [[0, 0, 0], [255, 255, 255]]
    class_names = ['background', 'body']
    select_class_indices = [class_names.index(
        cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # train_dataset = BodyDataset(x_dataset=train_dataset, dir=DATA_DIR)
    # valid_dataset = BodyDataset(x_dataset=valid_dataset, dir=DATA_DIR)

    # image, mask = train_dataset[0]
    # print(image.shape)
    # print(mask.shape)
    # visualize(
    #     original_image=image,
    #     ground_truth_mask=colour_code_segmentation(
    #         reverse_one_hot(mask), select_class_rgb_values),
    #     one_hot_encoded_mask=reverse_one_hot(mask)
    # )

    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    # could be None for logits or 'softmax2d' for multiclass segmentation
    ACTIVATION = 'sigmoid'

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    train_dataset = BodyDataset(
        IMAGES_DIR, MASKS_DIR, train_dataset, class_rgb_values=select_class_rgb_values, preprocessing=get_preprocessing(preprocessing_fn))

    valid_dataset = BodyDataset(
        IMAGES_DIR, MASKS_DIR, valid_dataset, class_rgb_values=select_class_rgb_values, preprocessing=get_preprocessing(preprocessing_fn))

    test_dataset = BodyDataset(
        IMAGES_DIR, MASKS_DIR, test_dataset, class_rgb_values=select_class_rgb_values, preprocessing=get_preprocessing(preprocessing_fn))

    # Get train and val data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

    # Set num of epochs
    EPOCHS = 16

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists('./best-model/best_model.pth'):
        model = torch.load('./best-model/best_model.pth', map_location=DEVICE)

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')
