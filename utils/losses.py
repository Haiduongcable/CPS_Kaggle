import numpy as np 
import time 
import torch 
import os 
import cv2
import torch.nn.functional as F


def cross_entropy_loss(pred, mask):
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    return wbce
# def dice_loss(logits, true, eps=1e-7):
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     print(true.shape)
#     # true = torch.unsqueeze(true, 1)
#     # print(true.shape)
#     print(true.squeeze(1))
    
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = F.one_hot(true, num_classes = -1)
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     # print(true_1_hot.shape)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return (1 - dice_loss)


def dice_loss(inputs, targets, ignore_index = 255, smooth=1, num_class = 21):
    '''
    inputs: max index B * H * W
    '''
    # np_target = targets.detach().cpu().numpy()
    # cv2.imwrite("visualize/log_target.jpg", np_target[0])
    # print(inputs.shape)
    # print(targets.shape)
    # inputs = inputs[:1,:,:]
    # targets = targets[:1,:,:]
    calculate_dice = 0.
    for value_class in range(num_class):
        if value_class == 0:
            logits_per_class = torch.where(targets == ignore_index, 255, inputs)
            target_per_class = torch.where(targets == ignore_index, 255, targets)
        else:
            logits_per_class = torch.where(targets == ignore_index, 240, inputs)
            target_per_class = torch.where(targets == ignore_index, 240, targets)
        logits_per_class = torch.where(logits_per_class == value_class, 1, 0)
        target_per_class = torch.where(target_per_class == value_class, 1, 0)
        # print(value_class)
        # print(logits_per_class[0].sum())
        # print(target_per_class[0].sum())
        logits_per_class = logits_per_class.view(-1)
        target_per_class = target_per_class.view(-1)
        intersection = (logits_per_class * target_per_class).sum()                            
        dice = (2.*intersection + smooth)/(logits_per_class.sum() + target_per_class.sum() + smooth)
        calculate_dice += dice
    dice_loss_score = 1 - calculate_dice/num_class
    return dice_loss_score

def structure_loss(pred, mask):
    '''
    Calculate Structure loss
    '''
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

def semi_ce_loss_1_class(inputs, targets,
                 conf_mask=True, threshold=0.6,
                 threshold_neg=0.0, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets/temperature_value, dim=1)
        
        # for positive
        targets_real_prob = F.softmax(targets, dim=1)
        
        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        if neg_label.shape[-1] != 1:
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], 1 - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label
          
        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return zero, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError
    
    
def semi_ce_loss_multi_class(inputs, targets,
                 conf_mask=True, threshold=0.6,
                 threshold_neg=0.0, temperature_value=1):
    # target => logit, input => logit
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets/temperature_value, dim=1)
        
        # for positive
        targets_real_prob = F.softmax(targets, dim=1)
        
        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        neg_label = torch.nn.functional.one_hot(torch.argmax(targets_prob, dim=1)).type(targets.dtype)
        if neg_label.shape[-1] != 21:
            neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
                                                           neg_label.shape[2], 21 - neg_label.shape[-1]]).cuda()),
                                  dim=3)
        neg_label = neg_label.permute(0, 3, 1, 2)
        neg_label = 1 - neg_label
          
        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return zero, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError



   
def semi_ce_loss_multi_class_target(inputs, targets,
                 conf_mask=True, threshold=0.6,
                 threshold_neg=0.0, temperature_value=1):
    # target => logit, input => logit
    # pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets/temperature_value, dim=1)
        
        # for positive
        targets_real_prob = F.softmax(targets, dim=1)
        
        weight = targets_real_prob.max(1)[0]
        # total_number = len(targets_prob.flatten(0))
        # boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
        #             "0.3~0.4", "0.4~0.5", "0.5~0.6",
        #             "0.6~0.7", "0.7~0.8", "0.8~0.9",
        #             "> 0.9"]

        # rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
        #         / total_number for i in range(1, 11)]

        # max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
        #             / weight.numel() for i in range(1, 11)]

        # pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        # pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)
          
        if not torch.any(mask):
            zero = torch.tensor(0., dtype=torch.float, device=targets.device)
            return zero
        else:
            positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")
            positive_loss_mat = positive_loss_mat * weight
            return positive_loss_mat[mask].mean()
    else:
        raise NotImplementedError

