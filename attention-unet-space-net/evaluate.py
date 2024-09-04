import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff



@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    all_labels = []
    all_preds = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true1 = mask_true.to(device=device, dtype=torch.long)
            mask_pred1 = net(image)
            mask_pred1 = (torch.sigmoid(mask_pred1) > 0.5).float()

            _, val_predicted1 = torch.max(mask_pred1.data, 1)

            all_labels.extend(mask_true1.cpu().numpy())
            all_preds.extend(val_predicted1.cpu().numpy())

            all_labels_arr = np.array(all_labels)
            all_preds_arr = np.array(all_preds)

            all_arr = all_preds_arr + 2 * all_labels_arr

            count_TP = np.count_nonzero(all_arr == 3)
            count_FP = np.count_nonzero(all_arr == 1)
            count_FN = np.count_nonzero(all_arr == 2)
            count_TN = np.count_nonzero(all_arr == 0)

            if count_TP + count_FP != 0:
                precision = count_TP / (count_TP + count_FP)
            else:
                precision = 0
            if count_TP + count_FN != 0:
                recall = count_TP / (count_TP + count_FN)
            else:
                recall = 0
            if (precision + recall) != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Compute Dice score for this batch
            if net.n_classes == 1:
                dice_score += dice_coeff(mask_pred1, mask_true1, reduce_batch_first=False)
            else:
                mask_true_one_hot = F.one_hot(mask_true1, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_one_hot = F.one_hot(mask_pred1.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)

    avg_precision = total_precision / max(num_val_batches, 1)
    avg_recall = total_recall / max(num_val_batches, 1)
    avg_f1 = total_f1 / max(num_val_batches, 1)

    net.train()
    return dice_score / max(num_val_batches, 1), avg_precision, avg_recall, avg_f1
