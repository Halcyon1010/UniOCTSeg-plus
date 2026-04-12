import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from OCTdataset.class2label import generate_all_taskonehot, onehot2layername

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def dice_score(preds, targets, smooth=1e-12):
    preds = (preds > 0.5).float()  # .astype(np.float32)
    preds = preds.view(preds.shape[0], preds.shape[1], -1).detach().cpu().numpy()
    targets = targets.view(targets.shape[0], targets.shape[1], -1).detach().cpu().numpy()
    # smooth = 1e-12
    smooth = 1

    m1 = preds
    m2 = targets
    intersection = (m1 * m2)
    score = (2. * (intersection.sum(2)) + smooth) / (m1.sum(2) + m2.sum(2) + smooth)
    return score


def get_non_zero_target(probs, targets):
    """
    Filters out channels where the target mask is entirely zero (empty).
    This prevents computing loss on classes that are not present in the current sample.
    
    Args:
        probs (Tensor): Prediction probabilities [B, C, H, W]
        targets (Tensor): Ground truth masks [B, C, H, W]
    
    Returns:
        valid_probs (Tensor): Filtered probabilities [B, N_valid, H, W]
        valid_targets (Tensor): Filtered targets [B, N_valid, H, W]
    """
    valid_probs = []
    valid_targets = []
    B, C, H, W = targets.shape
    
    # Iterate through channels to find non-empty targets
    for c in range(C):
        # Check if the channel is all zeros
        if torch.all(targets[:, c] == 0):
            continue
        else:
            valid_probs.append(probs[:, c])
            valid_targets.append(targets[:, c])
            
    if not valid_probs: # Handle case where all targets are zero
        return None, None

    valid_probs = torch.stack(valid_probs, dim=1)
    valid_targets = torch.stack(valid_targets, dim=1)
    
    return valid_probs, valid_targets

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        predict = predict.contiguous()
        target = target.contiguous()
        loss = self.criterion(predict, target.long())
        return loss

class SoftDiceLoss(nn.Module):
    
    """
    Computes the Soft Dice Loss for segmentation.
    Formula: 1 - (2 * Intersection + smooth) / (Union + smooth)
    """
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # Flatten inputs: [B, C, H, W] -> [B, C, N]
        probs = probs.view(*probs.shape[:2], -1)
        targets = targets.view(*targets.shape[:2], -1)
        
        # Calculate Dice Score
        intersection = (probs * targets).sum(2)
        union = probs.sum(2) + targets.sum(2)
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over channels, then average over batch
        score_channel_mean = dice_score.mean(1)
        score_batch_mean = score_channel_mean.mean()

        return 1 - score_batch_mean


class MultiBCELoss(nn.Module):
    
    """
    Multi-Label Binary Cross Entropy Loss.
    Only computes loss for classes present in the target (non-zero).
    """
    def __init__(self, ignore_index=None, num_classes=8, **kwargs):
        super(MultiBCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        predict = predict.contiguous()
        total_loss = []
        B, C, H, W = predict.shape

        for b in range(B):
            # Filter out empty classes for this sample
            probs, targets = get_non_zero_target(predict[b:b+1], target[b:b+1])
            
            if probs is not None:
                for c in range(probs.shape[1]):
                    # Compute BCE for each valid class channel
                    loss = self.criterion(probs[:, c:c+1], targets[:, c:c+1])
                    total_loss.append(loss)
        
        if len(total_loss) == 0:
            return torch.tensor(0.0, device=predict.device, requires_grad=True)

        total_loss = torch.stack(total_loss)
        return total_loss.mean()


class MultiSoftDiceLoss(nn.Module):
    """
    Multi-Label Soft Dice Loss.
    Similar to SoftDiceLoss but filters out empty target channels first.
    """
    def __init__(self, smooth=1.0):
        super(MultiSoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs_n, targets_n):
        loss = 0
        probs_n = probs_n.contiguous()
        B, C, H, W = probs_n.shape
        valid_samples = 0

        for b in range(B):
            # Filter empty targets per sample
            probs, targets = get_non_zero_target(probs_n[b:b+1], targets_n[b:b+1])
            
            if probs is None: 
                continue

            valid_samples += 1
            
            # Flatten: [1, C_valid, N]
            probs = probs.view(*probs.shape[:2], -1)
            targets = targets.view(*targets.shape[:2], -1)
            
            intersection = (probs * targets).sum(2)
            union = probs.sum(2) + targets.sum(2)
            
            score = (2. * intersection + self.smooth) / (union + self.smooth)
            
            # Average over valid channels
            score_channel_mean = score.mean(1)
            loss += (1 - score_channel_mean)

        if valid_samples > 0:
            loss = loss / valid_samples
        else:
            loss = torch.tensor(0.0, device=probs_n.device, requires_grad=True)
            
        return loss


class ProgressiveConsistencyLoss(nn.Module):
    """
    Progressive Consistency Loss for Multi-Task Learning.
    Dynamically updates target tasks based on loss thresholds and iteration counts.
    """
    def __init__(self, result_logger):
        super(ProgressiveConsistencyLoss, self).__init__()
        self.result_log = result_logger
        self.criterion = SoftDiceLoss()
        self.iter_count = 0
        self.collected_codes = []
        
        # Initialize task codes (assuming generate_all_taskonehot returns a list of codes)
        _, self.selected_codes, _ = generate_all_taskonehot()
        
        # Filter out single-task codes (sum == 1)
        self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]
        
        self.current_task_code = self.get_selected_codes()
        self.weight = 0.5
        
        # Thresholds based on task complexity (number of active tasks)
        self.base_thresholds = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]
        self._update_threshold()

    def _update_threshold(self):
        """Updates the loss threshold based on current task complexity."""
        task_complexity = int(torch.sum(self.current_task_code) - 1)
        # Ensure index is within bounds
        idx = min(task_complexity, len(self.base_thresholds) - 1)
        self.threshold = self.base_thresholds[idx]

    def collect_task_codes(self):
        self.collected_codes.append(self.current_task_code)

    def get_selected_codes(self):
        """
        Selects the next task code based on the highest occurrence/sum.
        If empty, re-generates the list.
        """
        if len(self.selected_codes) == 0:
            _, self.selected_codes, _ = generate_all_taskonehot()
            self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]

        # Calculate sum along axis 1 (assuming codes are list of arrays/lists)
        sum_up = np.sum(np.stack(self.selected_codes, axis=0), axis=1)
        best_idx = np.argmax(sum_up)
        out = self.selected_codes[best_idx]
        
        # Log and remove the selected code
        if self.result_log:
            self.result_log.write_content('consistency target: ', onehot2layername(out))
        
        del self.selected_codes[best_idx]
        return torch.tensor(out).long()

    def forward(self, ema_preds, preds, task_codes):
        self.current_task_code = self.current_task_code.to(preds.device)
        B, C, H, W = preds.shape
        preds = preds.contiguous()
        ema_preds = ema_preds.contiguous()

        # 1. Generate Pseudo Labels from EMA Model
        # Select channels where the input task code matches the current target task code
        pseudo_labels_list = []
        for c in range(C):
            if (task_codes[0, c] == self.current_task_code).all():
                pseudo_labels_list.append(ema_preds[:, c])
        
        if not pseudo_labels_list:
             # Fallback if no matching tasks found to avoid crash
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        pseudo_labels = torch.stack(pseudo_labels_list, dim=1).detach()
        pseudo_labels = (pseudo_labels >= 0.5).float() # Binarize

        # 2. Identify "Main" Sub-tasks
        # Logic: Find sub-tasks that are exactly 1 bit different (smaller) than current task
        current_sum = torch.sum(self.current_task_code)
        target_sum = current_sum - 1
        
        main_indices = []
        for c in range(C):
            code_c = task_codes[0, c]
            intersection = torch.sum(code_c * self.current_task_code)
            code_sum = torch.sum(code_c)
            
            if intersection == target_sum and code_sum == target_sum:
                main_indices.append(c)

        if not main_indices:
             return torch.tensor(0.0, device=preds.device, requires_grad=True)

        main_preds = torch.cat([preds[:, i:i+1] for i in main_indices], dim=1)
        main_task_codes = torch.stack([task_codes[:, i] for i in main_indices], dim=1)

        # 3. Compute Consistency Loss
        loss = 0
        for t in range(main_task_codes.shape[1]):
            # Find the "complementary" task (the missing bit)
            diff_code = (self.current_task_code - main_task_codes[0, t])
            # ReLU logic: ignore negative differences
            diff_code = torch.where(diff_code > 0, diff_code, torch.tensor(0, device=diff_code.device))

            # Find predictions corresponding to this complementary task
            cat_preds_list = [preds[:, c] for c in range(C) if (diff_code == task_codes[0, c]).all()]
            
            if cat_preds_list:
                cat_preds = torch.stack(cat_preds_list, dim=1)
                # Max fusion of main prediction and complementary prediction
                fused_pred = torch.max(torch.cat([main_preds[:, t:t+1], cat_preds], dim=1), keepdim=True, dim=1)[0]
                loss += self.criterion(fused_pred, pseudo_labels)

        loss /= main_task_codes.shape[1]
        weighted_loss = loss * self.weight

        # 4. Progressive Update Logic
        self.iter_count += 1
        if loss < self.threshold or self.iter_count >= 500:
            self.collect_task_codes()
            self.current_task_code = self.get_selected_codes()
            self._update_threshold()
            
            if self.result_log:
                self.result_log.write_content('current threshold: ', self.threshold)
            
            self.iter_count = 0

        return weighted_loss
    
def mask2onehot(mask, value_list):
    return np.array([mask == i for i in value_list]).astype(np.int8)

def get_non_zero_target_v3(probs_n, targets_n):
    probs = []
    targets = []
    B, C, _, H, W = probs_n.shape
    for c in range(0, C):
        if torch.all(torch.eq(targets_n[:,c], torch.tensor(0))):
            continue
        else:
            probs.append(probs_n[:,c:c+1])
            targets.append(targets_n[:,c:c+1])
    probs = torch.cat(probs, dim=1)
    targets = torch.cat(targets, dim=1)
    return probs, targets

class MultiSoftDiceLoss_v2(nn.Module):
    def __init__(self):
        super(MultiSoftDiceLoss_v2, self).__init__()
        self.criterion = SoftDiceLoss()

    def forward(self, probs_n, targets_n):
        total_loss = []
        probs_n = probs_n.contiguous()
        B, C, _, H, W = probs_n.shape
        for b in range(B):
            probs, targets = get_non_zero_target_v3(probs_n[b:b+1], targets_n[b:b+1])
            targets = torch.stack([torch.tensor(mask2onehot(targets[0:1, c].cpu().numpy(), [0, 1]), device=probs.device).permute(1,0,2,3) for c in range(targets.shape[1])], dim=1)            
            loss=[self.criterion(torch.softmax(probs[:, c], dim=1), targets[:, c]) for c in range(0, targets.shape[1])]
            total_loss.append(torch.stack(loss).mean())
        total_loss = torch.stack(total_loss)
        return total_loss.mean()
    

class Multi_Cross_entropy(nn.Module):
    def __init__(self, ignore_index=None, num_classes=8, **kwargs):
        super(Multi_Cross_entropy, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        #assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'
        total_loss = []
        predict = predict.contiguous()
        B, C, _, H, W = predict.shape

        for b in range(B):
            probs, targets = get_non_zero_target_v3(predict[b:b+1], target[b:b+1])
            loss=[self.criterion(probs[:, c], targets[:, c].to(device=probs.device).long()) for c in range(0, targets.shape[1])]
            total_loss.append(torch.stack(loss).mean())
        total_loss = torch.stack(total_loss)
        return total_loss.mean()
    

class Bidirection_consistency_loss(nn.Module):
    def __init__(self, result_loger):
        super(Bidirection_consistency_loss, self).__init__()
        self.result_log = result_loger
        self.criterion = SoftDiceLoss()
        self.iter_ = 0
        _, self.selected_codes, _ = generate_all_taskonehot()
        self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]
        
        self.Coarse_task_target = self.get_selected_codes()
        self.base_threshold = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]#[0.04, 0.02, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002]
        self.threshold = self.base_threshold[int(torch.sum(self.Coarse_task_target) - 1)]
        self.result_log.write_content('current threhold: ', self.threshold)

    def get_selected_codes(self):
        if len(self.selected_codes) != 0:
            sum_up = np.sum(np.stack(self.selected_codes, axis=0),axis=1)
            out = self.selected_codes[np.argmax(sum_up)]
            self.result_log.write_content('consistency target: ', onehot2layername(out))
            
            del self.selected_codes[np.argmax(sum_up)]
        else:
            _, self.selected_codes, _ = generate_all_taskonehot()
            self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]
            sum_up = np.sum(np.stack(self.selected_codes, axis=0), axis=1)
            out = self.selected_codes[np.argmax(sum_up)]
            self.result_log.write_content('consistency target: ', onehot2layername(out))
            del self.selected_codes[np.argmax(sum_up)]
        return torch.tensor(out).long()
    
    def Coarse2Fine(self, coarse_pred, main_coarse_pred, fine_target):
        coarse_prob = (coarse_pred>=0.5).float()
        main_coarse_prob = (main_coarse_pred >= 0.5).float()

        cut = torch.where((coarse_prob - main_coarse_prob)<0, 0, coarse_prob - main_coarse_prob).detach().cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for b in range(cut.shape[0]):
            for c in range(cut.shape[1]):
                cut[b,c] = cv2.erode(cut[b,c], kernel, iterations=1)
                cut[b,c] = cv2.dilate(cut[b,c], kernel, iterations=1)

        
        cut = torch.from_numpy(cut).to(coarse_pred.device)
        cut_prob = cut*coarse_pred
        loss = self.criterion(cut_prob, fine_target)

        return loss

    def Fine2Coarse(self, main_coarse_pred, fine_pred, coarse_target):
        # fine_prob = (fine_pred>=0.5).float()
        # main_coarse_prob = (main_coarse_pred >= 0.5).float()

        # fine_pred = fine_pred*fine_prob
        # main_coarse_prob = main_coarse_prob * main_coarse_prob
        cat_result = torch.max(torch.cat([main_coarse_pred, fine_pred], dim=1), keepdim=True, dim=1)[0]
        cat_copy = (cat_result>=0.5).float().detach().cpu().numpy()
        
        cat_result = self.hole_fill(cat_copy, cat_result)
        loss = self.criterion(cat_result, coarse_target)
        return loss

    def hole_fill(self, cat_copy, cat_result):
        cat_copy_ori = cat_copy.copy()
        for b in range(cat_copy.shape[0]):
            for c in range(cat_copy.shape[1]):
                contours, _ = cv2.findContours((cat_copy[b, c]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cat_copy[b, c], contours, -1, 1, thickness=cv2.FILLED)
        hole_img = cat_copy - cat_copy_ori
        if np.any(hole_img==1):
            cat_result[hole_img==1]=1
        return cat_result

    def forward(self, ema_preds, preds, task_codes):
        fine2coarse_loss_ = 0
        coarse2fine_loss_ = 0

        self.Coarse_task_target = self.Coarse_task_target.to(preds.device)

        preds = preds.contiguous()
        ema_preds = ema_preds.contiguous()
        C = preds.shape[1]

        Coarse_target_preds = torch.cat([preds[:,c:c+1] for c in range(C) if torch.equal(task_codes[0,c], self.Coarse_task_target) ], dim=1)
        Coarse_target_psudo = torch.cat([ema_preds[:,c:c+1] for c in range(C) if torch.equal(task_codes[0,c], self.Coarse_task_target) ], dim=1)

        Coarse_codes = [task_codes[0,c] for c in range(C) if torch.sum(task_codes[0,c]*self.Coarse_task_target) == (torch.sum(self.Coarse_task_target) - 1) and torch.sum(task_codes[0,c])== (torch.sum(self.Coarse_task_target) - 1)]
        Coarse_preds = torch.cat([preds[:,c:c+1] for t in Coarse_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)
        # Coarse_psudo = torch.cat([ema_preds[:,c:c+1] for t in Coarse_task_codes for c in range(C) if torch.equal(task_codes[0:1,c], t) ], dim=1)


        Fine_task_codes = [self.Coarse_task_target - t for t in Coarse_codes]
        Fine_preds = torch.cat([preds[:,c:c+1] for t in Fine_task_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)
        Fine_psudo = torch.cat([ema_preds[:,c:c+1] for t in Fine_task_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)

        # total_loss = 0
        for c in range(Coarse_preds.shape[1]):
            fine2coarse_loss = self.Fine2Coarse(Coarse_preds[:, c:c+1], Fine_preds[:, c:c+1], Coarse_target_psudo)
            coarse2fine_loss = self.Coarse2Fine(Coarse_target_preds, Coarse_preds[:, c:c+1], Fine_psudo[:, c:c+1])
            fine2coarse_loss_ += fine2coarse_loss
            coarse2fine_loss_ += coarse2fine_loss
            
        total_loss = (coarse2fine_loss_/Coarse_preds.shape[1]+fine2coarse_loss_/Coarse_preds.shape[1])/2
        self.iter_ += 1 
        if self.iter_>=500: #total_loss < self.threshold or 
            self.Coarse_task_target = self.get_selected_codes()
            self.threshold = self.base_threshold[int(torch.sum(self.Coarse_task_target) - 1)]
            self.result_log.write_content('current threhold: ', self.threshold)
            self.iter_ = 0
        return total_loss
    

def entropy_map_binary(p_fg, eps=1e-6):
    # p_fg: [B,C,H,W] foreground prob
    p = p_fg.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))  # [B,C,H,W]

def sigmoid_rampup(cur, rampup_length):
    # 简单 rampup（可替换成你现成的 ramps.sigmoid_rampup）
    if rampup_length == 0:
        return 1.0
    cur = max(0.0, min(1.0, cur / rampup_length))
    return float(torch.exp(torch.tensor(-5.0 * (1.0 - cur) ** 2)))


class MaskedSoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask=None):
        """
        pred/target: [B,1,H,W]
        mask: [B,1,H,W], values in {0,1} (或 soft weight 也可以)
        """
        pred = pred.float()
        target = target.float()

        if mask is not None:
            pred = pred * mask
            target = target * mask

        pred_f = pred.view(pred.size(0), -1)
        targ_f = target.view(target.size(0), -1)

        inter = (pred_f * targ_f).sum(dim=1)
        denom = pred_f.sum(dim=1) + targ_f.sum(dim=1)

        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()

    
def entropy_bin(p, eps=1e-8):
    p = p.clamp(eps, 1-eps)
    ent = -(p * p.log() + (1-p) * (1-p).log())   # [B,1,H,W]
    return ent / np.log(2.0)                     # normalize to [0,1]

    
class Bidirection_consistency_loss_semi(nn.Module):
    def __init__(self, result_loger):
        super(Bidirection_consistency_loss_semi, self).__init__()
        self.result_log = result_loger
        self.criterion = SoftDiceLoss()
        self.iter_ = 0
        _, self.selected_codes, _ = generate_all_taskonehot()
        self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]
        
        self.Coarse_task_target = self.get_selected_codes()
        self.base_threshold = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]#[0.04, 0.02, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002]
        self.threshold = self.base_threshold[int(torch.sum(self.Coarse_task_target) - 1)]
        self.result_log.write_content('current threhold: ', self.threshold)

    def get_selected_codes(self):
        if len(self.selected_codes) != 0:
            sum_up = np.sum(np.stack(self.selected_codes, axis=0),axis=1)
            out = self.selected_codes[np.argmax(sum_up)]
            self.result_log.write_content('consistency target: ', onehot2layername(out))
            
            del self.selected_codes[np.argmax(sum_up)]
        else:
            _, self.selected_codes, _ = generate_all_taskonehot()
            self.selected_codes = [code for code in self.selected_codes if np.sum(code) != 1]
            sum_up = np.sum(np.stack(self.selected_codes, axis=0), axis=1)
            out = self.selected_codes[np.argmax(sum_up)]
            self.result_log.write_content('consistency target: ', onehot2layername(out))
            del self.selected_codes[np.argmax(sum_up)]
        return torch.tensor(out).long()
    
    def Coarse2Fine(self, coarse_pred, main_coarse_pred, fine_target):
        coarse_prob = (coarse_pred>=0.5).float()
        main_coarse_prob = (main_coarse_pred >= 0.5).float()

        cut = torch.where((coarse_prob - main_coarse_prob)<0, 0, coarse_prob - main_coarse_prob).detach().cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for b in range(cut.shape[0]):
            for c in range(cut.shape[1]):
                cut[b,c] = cv2.erode(cut[b,c], kernel, iterations=1)
                cut[b,c] = cv2.dilate(cut[b,c], kernel, iterations=1)

        
        cut = torch.from_numpy(cut).to(coarse_pred.device)
        cut_prob = cut*coarse_pred
        loss = self.criterion(cut_prob, fine_target)

        return loss

    def Fine2Coarse(self, main_coarse_pred, fine_pred, coarse_target):

        cat_result = torch.max(torch.cat([main_coarse_pred, fine_pred], dim=1), keepdim=True, dim=1)[0]
        cat_copy = (cat_result>=0.5).float().detach().cpu().numpy()
        
        cat_result = self.hole_fill(cat_copy, cat_result)
        loss = self.criterion(cat_result, coarse_target)
        return loss

    def hole_fill(self, cat_copy, cat_result):
        cat_copy_ori = cat_copy.copy()
        for b in range(cat_copy.shape[0]):
            for c in range(cat_copy.shape[1]):
                contours, _ = cv2.findContours((cat_copy[b, c]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cat_copy[b, c], contours, -1, 1, thickness=cv2.FILLED)
        hole_img = cat_copy - cat_copy_ori
        if np.any(hole_img==1):
            cat_result[hole_img==1]=1
        return cat_result
    
    def _entropy_bin(self, p, eps=1e-8):
        """
        p: probability in [0,1], shape [B,1,...] or [B,C,...]
        return: normalized entropy in [0,1], same shape as p
        """
        p = p.clamp(eps, 1 - eps)
        ent = -(p * p.log() + (1 - p) * (1 - p).log())
        return ent / 0.6931471805599453  # log(2)

    def _ssl_unlabeled(self, ema_preds_u, preds_u, tau=0.35, mode="mse"):
        """
        Unlabeled-only consistency, focused on low-confidence (high-entropy) regions.
        ema_preds_u/preds_u: probs in [0,1], shape [Bu,C,...] (your case: [Bu, num_tasks, ...])
        """
        if preds_u.numel() == 0:
            return preds_u.sum()  # 0 with grad-safe

        ent = self._entropy_bin(ema_preds_u)           # [Bu,C,...]
        mask = (ent >= tau).float()                    # low-confidence region (high entropy)

        if mode == "mse":
            dist = (preds_u - ema_preds_u).pow(2)
        else:
            eps = 1e-8
            p = preds_u.clamp(eps, 1 - eps)
            q = ema_preds_u.clamp(eps, 1 - eps)
            dist = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())

        den = mask.sum().clamp_min(1.0)
        return (mask * dist).sum() / den


    def forward(self, ema_preds, preds, task_codes, labeled_bs=None, lam_ssl=0.1, tau_ssl=0.35):
        fine2coarse_loss_ = 0
        coarse2fine_loss_ = 0

        self.Coarse_task_target = self.Coarse_task_target.to(preds.device)

        preds = preds.contiguous()
        ema_preds = ema_preds.contiguous()
        C = preds.shape[1]

        Coarse_target_preds = torch.cat([preds[:,c:c+1] for c in range(C) if torch.equal(task_codes[0,c], self.Coarse_task_target) ], dim=1)
        Coarse_target_psudo = torch.cat([ema_preds[:,c:c+1] for c in range(C) if torch.equal(task_codes[0,c], self.Coarse_task_target) ], dim=1)

        Coarse_codes = [task_codes[0,c] for c in range(C) if torch.sum(task_codes[0,c]*self.Coarse_task_target) == (torch.sum(self.Coarse_task_target) - 1) and torch.sum(task_codes[0,c])== (torch.sum(self.Coarse_task_target) - 1)]
        Coarse_preds = torch.cat([preds[:,c:c+1] for t in Coarse_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)
        # Coarse_psudo = torch.cat([ema_preds[:,c:c+1] for t in Coarse_task_codes for c in range(C) if torch.equal(task_codes[0:1,c], t) ], dim=1)


        Fine_task_codes = [self.Coarse_task_target - t for t in Coarse_codes]
        Fine_preds = torch.cat([preds[:,c:c+1] for t in Fine_task_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)
        Fine_psudo = torch.cat([ema_preds[:,c:c+1] for t in Fine_task_codes for c in range(C) if torch.equal(task_codes[0,c], t) ], dim=1)

        # total_loss = 0
        for c in range(Coarse_preds.shape[1]):
            fine2coarse_loss = self.Fine2Coarse(Coarse_preds[:, c:c+1], Fine_preds[:, c:c+1], Coarse_target_psudo)
            coarse2fine_loss = self.Coarse2Fine(Coarse_target_preds, Coarse_preds[:, c:c+1], Fine_psudo[:, c:c+1])
            fine2coarse_loss_ += fine2coarse_loss
            coarse2fine_loss_ += coarse2fine_loss
    
        loss_bi = (coarse2fine_loss_/Coarse_preds.shape[1]+fine2coarse_loss_/Coarse_preds.shape[1])/2
        loss_ssl = preds.sum() * 0.0  # grad-safe zero
        if labeled_bs is not None:
            # assume batch layout: [labeled, unlabeled]
            preds_u = preds[labeled_bs:]
            ema_u = ema_preds[labeled_bs:]
            loss_ssl = self._ssl_unlabeled(ema_u, preds_u, tau=tau_ssl, mode="mse")
            
        total_loss = loss_bi + lam_ssl * loss_ssl

        self.iter_ += 1 
        if self.iter_>=500: #total_loss < self.threshold or 
            self.Coarse_task_target = self.get_selected_codes()
            self.threshold = self.base_threshold[int(torch.sum(self.Coarse_task_target) - 1)]
            self.result_log.write_content('current threhold: ', self.threshold)
            self.iter_ = 0
        return total_loss
    
