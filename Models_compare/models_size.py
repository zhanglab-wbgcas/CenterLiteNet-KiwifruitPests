import torch
from torchvision import models
from torch import nn
from thop import profile

# 修改模型最后一层
def modify_last_layer(model_name, new_num_classes):
    # 使用 torchvision 提供的直接从线上加载预训练权重的功能
    # 初始化模型并设置 pretrained=True 以自动下载并加载预训练权重
    if model_name in ['resnet50', 'resnet18', 'resnext50_32x4d', 'wide_resnet50_2', 'regnet_y_400mf', 'regnet_x_400mf', 'shufflenet_v2_x0_5']:
        convolutional_network = getattr(models, model_name)(pretrained=True)
        num_ftrs = convolutional_network.fc.in_features
        convolutional_network.fc = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['densenet121', 'efficientnet_b0', 'efficientnet_v2_s', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']:
        convolutional_network = getattr(models, model_name)(pretrained=True)
        num_ftrs = convolutional_network.classifier[-1].in_features
        convolutional_network.classifier = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['squeezenet1_0', 'squeezenet1_1']:
        convolutional_network = getattr(models, model_name)(pretrained=True)
        final_conv = nn.Conv2d(512, new_num_classes, kernel_size=(1, 1))
        convolutional_network.classifier[1] = final_conv
    elif model_name in ['googlenet', 'inception_v3']:
        convolutional_network = getattr(models, model_name)(pretrained=True)
        num_ftrs = convolutional_network.fc.in_features
        convolutional_network.fc = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['swin_t', 'vit_b_16']:
        convolutional_network = getattr(models, model_name)(pretrained=True)
        num_ftrs = convolutional_network.head.in_features
        convolutional_network.head = nn.Linear(num_ftrs, new_num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported for the custom head modification")
    return convolutional_network

# 模型列表
model_names = ["resnet18",   "shufflenet_v2_x0_5", "efficientnet_b0", "mobilenet_v2", "squeezenet1_1", "mnasnet0_5"]

# 定义类别数
new_num_classes = 10

# 遍历每个模型，计算 FLOPs 和参数数量
for model_name in model_names:
    print(f"\nEvaluating model: {model_name}")
    convolutional_network = modify_last_layer(model_name, new_num_classes)
    
    # 创建随机输入张量
    input_tensor = torch.randn(1, 3, 84, 84)
    
    # 计算 FLOPs 和参数
    flops, params = profile(convolutional_network, inputs=(input_tensor,), verbose=False)

    # 输出结果
    # print(f'Total parameters for {model_name}: {params}')
    # print(f'Total FLOPs for {model_name}: {flops}')
    # print(f'Total parameters calculated manually for {model_name}: {sum(p.numel() for p in convolutional_network.parameters())}')
    # 将参数转换为百万（M），FLOPs 转换为十亿（G）
    params_in_million = params / 1e6
    flops_in_giga = flops / 1e9

    # 输出结果
    print(f'Total parameters for {model_name}: {params_in_million:.2f}M')
    print(f'Total FLOPs for {model_name}: {flops_in_giga:.2f}G')
    print(f'Total parameters calculated manually for {model_name}: {sum(p.numel() for p in convolutional_network.parameters()) / 1e6:.2f}M')
