import torch

def pixel_level_prediction(features, seg_index):
    # 获取每个像素的预测类别
    logits = features
    y_pred = torch.argmax(logits, dim=1)
    
    # 假设 seg_index 形状与图像空间维度匹配,且它是一个二维张量
    h, w = seg_index.shape
    seg_index = seg_index.view(-1)  # 展平 seg_index 以便后续操作

    # 初始化一个与图像大小相同的张量来存储每个像素的预测标签
    pixel_predictions = torch.zeros((h * w,), dtype=torch.long, device=features.device)

    # 遍历每个超像素区域,将预测类别赋值给属于该区域的所有像素
    for idx in torch.unique(seg_index):
        # 找出属于当前超像素的所有像素的索引
        pixels_in_segment = (seg_index == idx).nonzero(as_tuple=False).squeeze()

        # 为这些像素赋予相应的预测类别
        pixel_predictions[pixels_in_segment] = y_pred[idx - 1]

    # 将 pixel_predictions 重塑回原始图像的形状
    pixel_predictions = pixel_predictions.view(h, w)

    return pixel_predictions