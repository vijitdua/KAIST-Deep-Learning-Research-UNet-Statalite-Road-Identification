import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from evaluate import evaluate
from unet import UNet, AttentionUNet
from utils.data_loading import BasicDataset, CarvanaDataset, MassachusettsDataset, M_Dataset

def test_model(model, device, test_loader, amp: bool = False):
    # Evaluation function
    dice_score, avg_precision, avg_recall, avg_f1 = evaluate(model, test_loader, device, amp)
    logging.info(f'Test Dice score: {dice_score}')
    logging.info(f'Test Precision: {avg_precision}')
    logging.info(f'Test Recall: {avg_recall}')
    logging.info(f'Test F1 Score: {avg_f1}')
    return dice_score, avg_precision, avg_recall, avg_f1

def load_model_and_test(checkpoint_path, test_img_dir, test_mask_dir, dataset, img_scale, batch_size, amp):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device=device)
    logging.info(f'Using device {device}')
    logging.info(f'Model loaded from {checkpoint_path}')

    # Helper function to load images and masks into lists
    def load_data(img_dir, mask_dir):
        imgs = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir) if img_name.endswith('.tif')]
        masks = [os.path.join(mask_dir, img_name.replace('.tif', '.tif')) for img_name in os.listdir(img_dir) if img_name.endswith('.tif')]
        return imgs, masks

    # Initialize lists for images and masks
    test_img_list = []
    test_mask_list = []

    # Process data based on the dataset selection
    if dataset in ['Vegas', 'both']:
        vegas_img_dir = os.path.join(test_img_dir, 'Vegas/test_images')
        vegas_mask_dir = os.path.join(test_mask_dir, 'Vegas/test_mask')
        vegas_imgs, vegas_masks = load_data(vegas_img_dir, vegas_mask_dir)
        test_img_list.extend(vegas_imgs)
        test_mask_list.extend(vegas_masks)

    if dataset in ['Shanghai', 'both']:
        shanghai_img_dir = os.path.join(test_img_dir, 'Shanghai/test_images')
        shanghai_mask_dir = os.path.join(test_mask_dir, 'Shanghai/test_mask')
        shanghai_imgs, shanghai_masks = load_data(shanghai_img_dir, shanghai_mask_dir)
        test_img_list.extend(shanghai_imgs)
        test_mask_list.extend(shanghai_masks)

    # Sort lists to ensure the pairs are correctly matched
    test_img_list.sort()
    test_mask_list.sort()

    # Create the dataset and DataLoader
    dataset = M_Dataset(test_img_list, test_mask_list, img_scale)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate the model
    test_model(model, device, test_loader, amp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, required=True, help='Load model from a .pth file')
    parser.add_argument('--test-imgs', type=str, required=True, help='Directory of test images')
    parser.add_argument('--test-masks', type=str, required=True, help='Directory of test masks')
    parser.add_argument('--dataset', type=str, required=True, choices=['Vegas', 'Shanghai', 'both'], help='Select dataset to test')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    args = parser.parse_args()

    load_model_and_test(
        checkpoint_path=args.load,
        test_img_dir=args.test_imgs,
        test_mask_dir=args.test_masks,
        dataset=args.dataset,
        img_scale=args.scale,
        batch_size=args.batch_size,
        amp=args.amp
    )
