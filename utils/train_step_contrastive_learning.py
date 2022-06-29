import torch 
import numpy as np 
import time 
import torch.nn.functional as F
from utils.data_augmentation import generate_unsup_data
from utils.contrastive_loss import compute_unsupervised_loss, compute_contra_memobank_loss
from config.config_contrastive_unreliable import config
from utils.process_data import label_onehot
# def generate_pseudolabel_and_predict_mask( branch = 1, augmentation = True, ratio_aug_cutmix = 0.3):
#     _,_,h,w = img_label.shape
#     #label image
#     pred_sup, rep_sup = model(img_label, step=branch)
#     prob_sup = F.softmax(pred_sup, dim=1)
    
#     #unlabel image
#     pred_unsup, rep_unsup = model(image_unlabel, step=branch)
#     prob_unsup = F.softmax(pred_unsup, dim=1)
   
#     pred_u_large = F.interpolate(
#                 prob_unsup, (h, w), mode="bilinear", align_corners=True
#     )
    
#     pred_l_large = F.interpolate(
#                 prob_sup, (h, w), mode="bilinear", align_corners=True
#     )
    
#     logits_u, label_u = torch.max(pred_u_large, dim=1)
#     if augmentation and np.random.uniform(0, 1) < ratio_aug_cutmix:
#         image_unlabel, label_u, logits_u = generate_unsup_data(
#                     image_unlabel,
#                     label_u.clone(),
#                     logits_u.clone(),
#                     mode='cutmix')
#     represent_all = torch.cat((rep_sup, rep_unsup))
#     prob_all = torch.cat((prob_sup, prob_unsup))
#     return image_unlabel, label_u, logits_u, pred_l_large, prob_all, represent_all



def train_step(model,img_label, gts, image_unlabel,\
                current_epoch,supervised_criterion,\
                memory_bank, queue_ptrlis, queue_size,\
                trained_model = 'left'):
    batch_size, h,w = gts.shape
    if trained_model == 'left':
        step_student = 1
        step_teacher = 2
    else:
        step_student = 2
        step_teacher = 1
    #generate pseudo label and augmentation from teacher model (step_teacher)
    with torch.no_grad():
        pred_u_teacher,_ = model(image_unlabel, step = step_teacher)
        
        # pred_u_teacher = F.interpolate(
        #     pred_u_teacher, (h, w), mode="bilinear", align_corners=True
        # )
        pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
        logits_u, label_u = torch.max(pred_u_teacher, dim=1)
        if np.random.uniform(0, 1) < config.ratio_aug_cutmix:
            image_unlabel, label_u, logits_u = generate_unsup_data(
                image_unlabel,
                label_u.clone(),
                logits_u.clone(),
                mode='cutmix')
    #Model student generate mask (step_student)
    pred_l_student, rep_l_student = model(img_label, step = step_student)
    pred_u_student, rep_u_student = model(image_unlabel, step = step_student)
    rep_all_student = torch.cat((rep_l_student,rep_u_student))
    pred_l_large_student = pred_l_student
    pred_u_large_student = pred_u_student
    # pred_l_large_student = F.interpolate(
    #         pred_l_student, size=(h, w), mode="bilinear", align_corners=True)
    # pred_u_large_student = F.interpolate(
    #         pred_u_student, size=(h, w), mode="bilinear", align_corners=True)
    sup_loss = supervised_criterion(pred_l_large_student, gts)
    
    #generate pseudo label from image augmentation model teacher (step_teacher)
    with torch.no_grad():
        pred_l_teacher, rep_l_teacher = model(img_label, step = step_teacher)
        pred_u_teacher, rep_u_teacher = model(image_unlabel, step = step_teacher)
        rep_all_teacher = torch.cat((rep_l_teacher, rep_u_teacher))
        prob_l_teacher = F.softmax(pred_l_teacher, dim=1)
        prob_l_teacher =  F.interpolate(prob_l_teacher, size=rep_l_teacher.shape[2:], mode="bilinear", align_corners=True)
        prob_u_teacher = F.softmax(pred_u_teacher, dim=1)
        prob_u_teacher =  F.interpolate(prob_u_teacher, size=rep_l_teacher.shape[2:], mode="bilinear", align_corners=True)
        pred_u_large_teacher = pred_u_teacher
        # pred_u_large_teacher = F.interpolate(
        #             pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True)
    #Calculate unsupervised loss based reliable pseudo label 
    drop_percent = config.drop_percent
    percent_unreliable = (100 - drop_percent) * (1 - current_epoch / config.total_epoch)
    drop_percent = 100 - percent_unreliable
    unsup_loss = (compute_unsupervised_loss(pred_u_large_student, label_u.clone(),\
                        drop_percent,pred_u_large_teacher.detach(),)\
                        * config.unsupervised_loss_weight)

    #Prepare for contrastive loss
    alpha_t = config.low_entropy_threshold * (1 - current_epoch/config.total_epoch)
    #generate entropy from teacher 
    with torch.no_grad():
        prob = torch.softmax(pred_u_large_teacher, dim = 1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        low_thresh = np.percentile(entropy[label_u != 255].cpu().numpy().flatten(), alpha_t)
        low_entropy_mask = (entropy.le(low_thresh).float() * (label_u != 255).bool())
        high_thresh = np.percentile(entropy[label_u != 255].cpu().numpy().flatten(), 100 - alpha_t,)
        high_entropy_mask = (entropy.ge(high_thresh).float() * (label_u != 255).bool())
        low_mask_all = torch.cat(((gts.unsqueeze(1) != 255).float(),\
                                    low_entropy_mask.unsqueeze(1),))
        #downsample
        low_mask_all = F.interpolate(
                        low_mask_all, size=rep_l_student.shape[2:], mode="nearest")
        
        #negative high entropy 
        high_mask_all = torch.cat(((gts.unsqueeze(1) != 255).float(), high_entropy_mask.unsqueeze(1)))
        #downsample 
        high_mask_all = F.interpolate(
                        high_mask_all, size=rep_l_student.shape[2:], mode="nearest") 
        # down sample and concat
        label_l_small = F.interpolate(
                        label_onehot(gts, config.num_classes),
                        size=rep_l_student.shape[2:],
                        mode="nearest",)
        label_u_small = F.interpolate(
                        label_onehot(label_u, config.num_classes),
                        size=rep_l_student.shape[2:],
                        mode="nearest",)
    # print(rep_all_student.shape)
    # print(label_l_small.shape)
    # print(label_u_small.shape)
    # print(prob_l_teacher.shape)
    # print(prob_u_teacher.shape)
    # print(low_mask_all.shape)
    # print(high_mask_all.shape)
    # print(rep_all_teacher.shape)
    new_keys, contrastive_loss = compute_contra_memobank_loss(
                        rep_all_student,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        config,
                        memory_bank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach())
    return sup_loss, unsup_loss, contrastive_loss