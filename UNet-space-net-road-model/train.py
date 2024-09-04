import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, MassachusettsDataset, M_Dataset
from utils.dice_score import dice_loss

# python train.py --amp --batch-size 4 --learning-rate 0.0001
# python train.py --batch-size 4
# python train.py --batch-size 1 --learning-rate 0.000000000001
# python train.py --e 25 --batch-size 8 --learning-rate 0.0001 --s 1
# python train.py --e 25 --batch-size 4 --learning-rate 0.0001
# python train.py --e 25 --batch-size 8 --learning-rate 0.00000001
# python train.py --e 25 --batch-size 4 --learning-rate 0.00000001

# dir_img = Path('/data/kiss2024/Massachusetts_Roads_Dataset/tiff/train/')
# dir_mask = Path('/data/kiss2024/Massachusetts_Roads_Dataset/tiff/train_labels/')
# dir_img = Path('/home/kiss2024/data/Carvana/imgs/')
# dir_mask = Path('/home/kiss2024/data/Carvana/masks/')
dir_img = Path('/home/kiss2024/data/space-net-8-bit/')
dir_mask = Path('/home/kiss2024/data/space-net-8-bit/')
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


# Seed setting functions
def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    # try:
    #     dataset = MassachusettsDataset(dir_img, dir_mask, img_scale)
    #     # dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     print('Creating BasicDataset for {}'.format(str(dir_img)))
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    if 'Carvana' in str(dir_img):
        print('Creating BasicDataset for {}'.format(str(dir_img)))
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif 'Massachusetts' in str(dir_img):
        train_img_dir = os.path.join(str(dir_img), 'train')
        train_mask_dir = os.path.join(str(dir_img), 'train_labels')
        val_img_dir = os.path.join(str(dir_img), 'test')
        val_mask_dir = os.path.join(str(dir_img), 'test_labels')

        # Following lists will contain will contain full directories to each image and mask
        train_img_list = []
        train_mask_list = []
        val_img_list = []
        val_mask_list = []

        for img_name in os.listdir(train_img_dir):
            train_img_list.append(os.path.join(train_img_dir, img_name))
            train_mask_list.append(os.path.join(train_mask_dir, img_name[:-1]))

        for img_name_val in os.listdir(val_img_dir):
            val_img_list.append(os.path.join(val_img_dir, img_name_val))
            val_mask_list.append(os.path.join(val_mask_dir, img_name_val[:-1]))

        n_train = len(train_img_list)
        n_val = len(val_img_list)

        train_set = M_Dataset(train_img_list, train_mask_list, img_scale)
        val_set = M_Dataset(val_img_list, val_mask_list, img_scale)

    # Space-net dataset configurable for combined or singular dataset training
    elif 'space-net' in str(dir_img):
        # Configuration: Set to 'both', 'one', or 'two' for dataset selection
        train_on = 'both'

        # Paths for location "one"
        one_img_dir = os.path.join(str(dir_img), 'Vegas/images')
        one_mask_dir = os.path.join(str(dir_img), 'Vegas/mask_tif')
        one_val_img_dir = os.path.join(str(dir_img), 'Vegas/test_images')
        one_val_mask_dir = os.path.join(str(dir_img), 'Vegas/test_mask')

        # Paths for location "two"
        two_img_dir = os.path.join(str(dir_img), 'Shanghai/images')
        two_mask_dir = os.path.join(str(dir_img), 'Shanghai/mask_tif')
        two_val_img_dir = os.path.join(str(dir_img), 'Shanghai/test_images')
        two_val_mask_dir = os.path.join(str(dir_img), 'Shanghai/test_mask')

        # Initialize lists
        train_img_list = []
        train_mask_list = []
        val_img_list = []
        val_mask_list = []

        # Helper function to load images and masks
        def load_data(img_dir, mask_dir, img_list, mask_list):
            for img_name in os.listdir(img_dir):
                img_list.append(os.path.join(img_dir, img_name))
                mask_name = img_name.replace('.tif', '.tif')  # Modify if necessary
                mask_list.append(os.path.join(mask_dir, mask_name))

        # Conditional data loading based on configuration
        if train_on in ['both', 'one']:
            load_data(one_img_dir, one_mask_dir, train_img_list, train_mask_list)
            load_data(one_val_img_dir, one_val_mask_dir, val_img_list, val_mask_list)

        if train_on in ['both', 'two']:
            load_data(two_img_dir, two_mask_dir, train_img_list, train_mask_list)
            load_data(two_val_img_dir, two_val_mask_dir, val_img_list, val_mask_list)

        # Combine datasets from selected locations
        n_train = len(train_img_list)
        n_val = len(val_img_list)
        train_set = M_Dataset(train_img_list, train_mask_list, img_scale)
        val_set = M_Dataset(val_img_list, val_mask_list, img_scale)

    # Space-net dataset regular
    # elif 'space-net' in str(dir_img):
    #     train_img_dir = os.path.join(str(dir_img), 'Shanghai/images')
    #     train_mask_dir = os.path.join(str(dir_img), 'Shanghai/mask_tif')
    #     val_img_dir = os.path.join(str(dir_img), 'Shanghai/test_images')
    #     val_mask_dir = os.path.join(str(dir_img), 'Shanghai/test_mask')
    #
    #     # Following lists will contain full directories to each image and mask
    #     train_img_list = []
    #     train_mask_list = []
    #     val_img_list = []
    #     val_mask_list = []
    #
    #     for img_name in os.listdir(train_img_dir):
    #         train_img_list.append(os.path.join(train_img_dir, img_name))
    #         # Correct the mask file name handling
    #         mask_name = img_name.replace('.tif', '.tif')
    #         train_mask_list.append(os.path.join(train_mask_dir, mask_name))
    #
    #     for img_name_val in os.listdir(val_img_dir):
    #         val_img_list.append(os.path.join(val_img_dir, img_name_val))
    #         # Correct the mask file name handling
    #         mask_name_val = img_name_val.replace('.tif', '.tif')
    #         val_mask_list.append(os.path.join(val_mask_dir, mask_name_val))
    #
    #     n_train = len(train_img_list)
    #     n_val = len(val_img_list)
    #
    #     train_set = M_Dataset(train_img_list, train_mask_list, img_scale)
    #     val_set = M_Dataset(val_img_list, val_mask_list, img_scale)

    # # Space-net dataset, with mixed data and random splitting (not wanted anymore)
    # elif 'space-net' in str(dir_img):
    #     vegas_img_dir = os.path.join(str(dir_img), 'Vegas/images')
    #     vegas_mask_dir = os.path.join(str(dir_img), 'Vegas/mask')
    #     shanghai_img_dir = os.path.join(str(dir_img), 'Shanghai/images')
    #     shanghai_mask_dir = os.path.join(str(dir_img), 'Shanghai/mask')
    #
    #     # Following lists will contain full directories to each image and mask
    #     img_list = []
    #     mask_list = []
    #
    #     for img_dir, mask_dir in [(vegas_img_dir, vegas_mask_dir), (shanghai_img_dir, shanghai_mask_dir)]:
    #         for img_name in os.listdir(img_dir):
    #             img_list.append(os.path.join(img_dir, img_name))
    #             mask_name = img_name.replace('.tif', '.png')
    #             mask_list.append(os.path.join(mask_dir, mask_name))
    #
    #     dataset = M_Dataset(img_list, mask_list, img_scale)
    #     n_val = int(len(dataset) * val_percent)
    #     n_train = len(dataset) - n_val
    #     train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                # print('image shape : ', images.shape)
                # print('mask shape : ', true_masks.shape)
                # print('image unique values : ', torch.unique(images))
                # print('mask unique values : ', torch.unique(true_masks))

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, avg_precision, avg_recall, avg_f1 = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info(f'Validation Dice score: {val_score}')
                        logging.info(f'Validation Precision: {avg_precision}')
                        logging.info(f'Validation Recall: {avg_recall}')
                        logging.info(f'Validation F1 Score: {avg_f1}')

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'validation Precision': avg_precision,
                                'validation Recall': avg_recall,
                                'validation F1': avg_f1,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Set seed
    seed = args.seed
    set_seed(seed)
    rand_seed = init_random_seed(seed)
    print('Seed : {}'.format(rand_seed))


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except KeyboardInterrupt:
        pass
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Co