import torch

def performance(predict_labels, gt_labels, class_num):
    predict_labels = torch.argmax(predict_labels, dim=1)
    
    matrix = torch.zeros((class_num, class_num), dtype=torch.int64)
    
    for prediction, ground_truth in zip(predict_labels, gt_labels):
        if ground_truth > 0:
            matrix[prediction - 1, ground_truth - 1] += 1
    
    OA = torch.diag(matrix).sum() / matrix.sum()
    
    AC_list = torch.diag(matrix) / matrix.sum(dim=0)
    
    AA = AC_list.mean()
    
    PE = torch.sum(matrix, dim=0) @ torch.sum(matrix, dim=1) / matrix.sum()**2
    
    PA = torch.diag(matrix).sum() / matrix.sum()
    
    Kappa = (PA - PE) / (1 - PE)
    
    return OA.item(), AA.item(), Kappa.item(), AC_list.tolist()

def acc(pixel_pred, label):
    pixel_pred = pixel_pred.cpu()
    label = label.cpu()
    acc = torch.eq(pixel_pred, label).float().mean()
    return acc.item()