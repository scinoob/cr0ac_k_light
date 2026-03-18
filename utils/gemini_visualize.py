import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import sys
# 将项目根目录添加到Python路径
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（根据你的目录结构调整，这里假设utils目录的上级是根目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入Python路径
sys.path.append(project_root)

from models.crack_net import CrackSegmentationNetV2,CrackSegmentationNet
from utils.gradcam import GradCAM
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_test_transform(target_size=512):
    """简单的测试期数据预处理"""
    return A.Compose([
        A.Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def plot_all_layers_cam(image_tensor, cams_dict, pred_mask, save_path):
    """将原图、预测结果以及各层热力图绘制在同一张画板上"""
    # 还原图像用于显示
    img_show = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_show = img_show * 0.5 + 0.5  # 反归一化
    img_show = np.clip(img_show, 0, 1)

    # 预测的掩码二值化
    pred_bin = (pred_mask > 0.5).astype(np.uint8)

    # 计算排版：原图 + 预测 + 所有监控的层
    total_plots = 2 + len(cams_dict)
    cols = 4
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # 1. 绘制原图
    axes[0].imshow(img_show)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # 2. 绘制预测结果
    axes[1].imshow(img_show)
    axes[1].imshow(pred_bin, cmap='jet', alpha=0.5)
    axes[1].set_title('Predicted Mask Overlay', fontsize=14)
    axes[1].axis('off')

    # 3. 绘制各层特征热力图
    plot_idx = 2
    for layer_name, cam in cams_dict.items():
        # CAM 缩放到原图大小
        cam_resized = cv2.resize(cam, (img_show.shape[1], img_show.shape[0]))
        
        axes[plot_idx].imshow(img_show)
        axes[plot_idx].imshow(cam_resized, cmap='jet', alpha=0.55)
        
        # 标注属于哪种类型的层
        if 'stage1' in layer_name or 'stage2' in layer_name:
            layer_desc = "(CNN Local Features)"
        elif 'stage3' in layer_name or 'stage4' in layer_name:
            layer_desc = "(Mamba Global Topology)"
        else:
            layer_desc = "(Decoder Fusion)"
            
        axes[plot_idx].set_title(f'{layer_name}\n{layer_desc}', fontsize=12)
        axes[plot_idx].axis('off')
        plot_idx += 1

    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"热力图分析已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="CrackSegmentationNetV2 逐层注意力可视化")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--image_path', type=str, required=True, help='测试图像路径')
    parser.add_argument('--input_size', type=int, default=512, help='输入尺寸')
    parser.add_argument('--save_dir', type=str, default='./cam_analysis', help='保存目录')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化 CrackSegmentationNetV2
    print("加载模型...")
    model = CrackSegmentationNetV2(
        in_channels=3, 
        num_classes=1, 
        base_channels=96, 
        input_size=args.input_size, 
        use_gbc=True
    )

    # model = CrackSegmentationNet(
    #     in_channels=3, 
    #     num_classes=1, 
    #     base_channels=96, 
    #     input_size=args.input_size, 
    #     use_gbc=True
    # )
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # 2. 定义需要观察的层 (映射到 CrackSegmentationNetV2 中的真实变量名)
    target_layers = [
        'stage1', # 浅层 PConv (提取边缘/高频信息)
        'stage2', # 浅层 PConv 
        'stage3', # 深层 AGM Mamba (提取拓扑/连通性)
        'stage4', # 深层 AGM Mamba (最深全局感受野)
        'fuse3',  # 解码器初期融合
        'fuse2',  # 解码器中期融合
        'fuse1'   # 解码器末期融合
    ]

    # target_layers = [
    #     'encoder1', 
    #     'encoder2', 
    #     'encoder3', 
    #     'encoder4', 
    #     'decoder.bottleneck'  # V1的解码器最深处
    # ]

    # 利用代码库现有的 GradCAM 工具
    gradcam = GradCAM(model, target_layers)

    # 3. 准备图像
    print(f"处理图像: {args.image_path}")
    orig_img = cv2.imread(args.image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    transform = get_test_transform(target_size=args.input_size)
    augmented = transform(image=orig_img)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # 4. 生成热力图
    print("计算梯度与热力图中...")
    # 预测并计算相对于网络输出的梯度
    cams_dict = gradcam.generate(input_tensor)
    
    # 提取最终的模型预测概率图（用于对比）
    with torch.no_grad():
        output = model(input_tensor)
        pred_prob = output.squeeze().cpu().numpy()

    # 5. 绘图与保存
    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.save_dir, f'{img_name}_layer_analysis.png')
    
    plot_all_layers_cam(input_tensor, cams_dict, pred_prob, save_path)
    
    # 清理钩子
    gradcam.remove_hooks()

if __name__ == '__main__':
    main()