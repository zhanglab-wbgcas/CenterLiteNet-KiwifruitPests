
import copy
from pathlib import Path
import random
from statistics import mean
from torchvision import models
import argparse

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import os
import torch
from thop import profile
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

random_seed = 4
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--save-path', default='./save/pre_match')
parser.add_argument('--gpu', default='0')
parser.add_argument('--pretrain_model', default='squeezenet1_1')
parser.add_argument('--imagesize', default='84')
parser.add_argument('--weight_decay', default='5e-4')
parser.add_argument('--feature_dim', default='20')
parser.add_argument('--lr', type=float, default='0.001', help='L2 regularization strength')


args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)


# Train the model
start_time = time.time()

n_way = args.way
n_shot =args.shot
n_query = args.query


# 检测是否有 CUDA 设备可用, 优先使用 CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_workers = 12
import sys
sys.path.append('/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro')
import easyfsl
from easyfsl.datasets import KIWI
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import ParameterSampler


n_tasks_per_epoch = 100  #500  200
n_validation_tasks = 50  #100
ImageSize = int(args.imagesize)

transform = transforms.Compose([
        transforms.RandomResizedCrop(ImageSize, scale=(0.8, 1.0)),  # 随机裁剪并调整到给定尺寸
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomVerticalFlip(),  # 垂直翻转
        transforms.RandomRotation(15),  # 随机旋转 ±15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # 高斯模糊
        transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# Instantiate the datasets
train_set = KIWI(split="train", training=True, transform=transform)
val_set = KIWI(split="val", training=False, transform=transform)

# Those are special batch samplers that sample few-shot classification tasks with a pre-defined shape
train_sampler = TaskSampler(
    train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
)
val_sampler = TaskSampler(
    val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)

# Finally, the DataLoader. We customize the collate_fn so that batches are delivered
# in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)

feature_dimension = args.feature_dim
from easyfsl.methods import MatchingNetworks,FewShotClassifier   #BDCSPN, FewShotClassifier
# 获取模型名称
model_name = args.pretrain_model


def modify_last_layer(model_name, new_num_classes):
    
    weights_dir = '/public/home/ZXJ_liukc/few_shot_dir/01multi_weight_pro/pth_dir/'
    weight_path = os.path.join(weights_dir, f'{model_name}.pth')

    # 检查权重文件是否存在，如果不存在则下载
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file for {model_name} not found at {weight_path}")

    # 初始化模型并加载预训练权重
    
    if model_name in ['resnet50', 'resnet18', 'resnext50_32x4d', 'wide_resnet50_2', 'regnet_y_400mf', 'regnet_x_400mf', 'shufflenet_v2_x0_5']:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path))
        num_ftrs = convolutional_network.fc.in_features
        convolutional_network.fc = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['densenet121', 'efficientnet_b0', 'efficientnet_v2_s', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_large']:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path))        
        num_ftrs = convolutional_network.classifier[-1].in_features
        convolutional_network.classifier = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['mobilenet_v3_small']:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path))        
        num_ftrs = convolutional_network.classifier[3].in_features
        convolutional_network.classifier = nn.Linear(num_ftrs, new_num_classes)
        
    elif model_name in ['squeezenet1_0',"squeezenet1_1"]:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path)) 
        final_conv = nn.Conv2d(512, new_num_classes, kernel_size=(1, 1))  # Adjust the number of output channels
        convolutional_network.classifier[1] = final_conv
    

    elif model_name in ['googlenet', 'inception_v3']:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path))
        num_ftrs = convolutional_network.fc.in_features
        convolutional_network.fc = nn.Linear(num_ftrs, new_num_classes)
    elif model_name in ['swin_t', 'vit_b_16']:
        convolutional_network = getattr(models, model_name)(pretrained=False).to(DEVICE) ##因为 pretrained=True 会尝试从网络下载预训练权重
        convolutional_network.load_state_dict(torch.load(weight_path))
        num_ftrs = convolutional_network.head.in_features
        convolutional_network.head = nn.Linear(num_ftrs, new_num_classes)
    
    else:
        raise ValueError(f"Model {model_name} is not supported for the custom head modification")
    return convolutional_network

convolutional_network= modify_last_layer(model_name, 20)

# Use a dummy input to compute FLOPs
input_tensor = torch.randn(1, 3, 84, 84)

# Calculate FLOPs and Parameters
flops, params = profile(convolutional_network, inputs=(input_tensor, ), verbose=False)

print(f'Total change_network parameters: {sum(p.numel() for p in convolutional_network.parameters())}')
print(f'Total change_network FLOPs: {flops}')


LOSS_FUNCTION = nn.CrossEntropyLoss()

n_epochs = 50  #200
scheduler_milestones = [30, 40]  #[120, 160]
scheduler_gamma = 0.1  #0.5

tb_logs_dir= args.save_path
weight_decay = float(args.weight_decay)  #1e-4  1e-3


tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
from easyfsl.utils import evaluate,evaluate_on_one_task,evaluate_loss,evaluate_ontest,visual_ontest

def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer
):
    all_loss = []
    all_acc = []

    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            classification_scores = model(query_images.to(DEVICE))

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
                     
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            correct_predictions, total_predictions = evaluate_on_one_task(
                model,
                support_images.to(DEVICE),
                support_labels.to(DEVICE),
                query_images.to(DEVICE),
                query_labels.to(DEVICE),
            )
            batch_accuracy = correct_predictions / total_predictions
            all_acc.append(batch_accuracy)

            tqdm_train.set_postfix(loss=mean(all_loss), accuracy=mean(all_acc))

    return mean(all_acc),mean(all_loss)


import matplotlib.pyplot as plt
import json
import os
import math

def save_results_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_and_save_results(data, labels, title, filename):
    plt.figure(figsize=(10, 5))
    for label, values in zip(labels, data):
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def train_and_validate(few_shot_classifier, train_loader, val_loader,learning_rates,n_epochs, visualize=False):
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    print(f"Training with learning rate: {learning_rates}")
    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rates, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )


    results = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_state = few_shot_classifier.state_dict()
    best_loss = float('inf') # 初始化为正无穷   
    best_acc = 0  
    best_validation_accuracy = 0 


    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        # start_time = time.time()

        average_acc,average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy,validation_loss = evaluate_loss(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            print("Ding ding ding! We found a new best model with lower validation loss!")


        # 检查是否有提升
        # if average_acc > best_acc:
        #     best_acc = average_acc 
        #     # best_state = copy.deepcopy(few_shot_classifier.state_dict())
        #     no_improvement_count = 0  # 重置计数器
        #     print("Ding ding ding! We found a new best model!")
        # else:
        #     no_improvement_count += 1  # 增加计数器

        # # 检查连续5个epoch没有提升的情况
        # if no_improvement_count >= 5:
        #     print("No improvement in the last 5 epochs. Skipping to the next parameter set.")
        #     break


        results["train_loss"].append(average_loss)
        results["train_accuracy"].append(average_acc)
        results["val_loss"].append(validation_loss)
        results["val_accuracy"].append(validation_accuracy)

        
    train_scheduler.step()

    
    few_shot_classifier.load_state_dict(best_state)

    #Evaluation
    #First step: we fetch the test data.
    n_test_tasks = 100  #1000

    test_set = KIWI(split="test", training=False)
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    test_accuracy, all_true_labels, all_predicted_probs, test_metrics = evaluate_ontest(
    few_shot_classifier, test_loader, device=DEVICE, tqdm_prefix="Test")
    print(f"Average roc test accuracy: {(100 * test_accuracy):.2f} %")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_metrics['precision']}")
    print(f"Test Recall: {test_metrics['recall']}")
    print(f"Test F1 Score: {test_metrics['f1_score']}")

    # # 保存 ROC 曲线图像
    roc_filename = f'roc_curve_kiwi_match_{args.pretrain_model}_{n_way}_{n_shot}.png'
    plot_roc_curve_single(all_true_labels, all_predicted_probs, save_path=roc_filename)
    #保存roc结果
    save_data_path = f'roc_curve_kiwi_match_{args.pretrain_model}_{n_way}_{n_shot}.npz'
    np.savez(save_data_path, true_labels=all_true_labels, predicted_probs=all_predicted_probs)

    
    # 可视化 t-SNE
    if visualize:
        tsne_filename = f'tsne_visualization_kiwi_match_{args.pretrain_model}_{n_way}_{n_shot}.png'
        visual_ontest(few_shot_classifier, test_loader, device=DEVICE, tqdm_prefix="Test", visualize=True, save_path=tsne_filename)
    
    return results,test_accuracy
    
from sklearn.metrics import roc_curve, auc

def plot_roc_curve_single(true_labels, predicted_probs, save_path):
    """
    Plot and save a single ROC curve by averaging across all classes.
    Args:
        true_labels: True labels for the test data
        predicted_probs: Predicted probabilities from the model
        save_path: Path to save the ROC curve plot
    """
    # Binarize the labels (for multiclass, we calculate ROC for each class)
    true_labels_bin = np.eye(len(np.unique(true_labels)))[true_labels]

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = true_labels_bin.shape[1]

    # Collect the FPR, TPR for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], np.array(predicted_probs)[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(), np.array(predicted_probs).ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, color='b', label=f"Micro-average (AUC = {roc_auc_micro:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(save_path)
    plt.close()

        
few_shot_classifier =MatchingNetworks(
        convolutional_network, 
        feature_dimension=int(args.feature_dim), 
    )
total_params = sum(p.numel() for p in few_shot_classifier.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# 训练并评估模型
results,test_acc = train_and_validate(few_shot_classifier, train_loader, val_loader, args.lr,n_epochs=50,visualize=True)
print(f"Parameters of this epoch: {params}")

save_results_to_json(results, f'match_step_KIWI_{args.pretrain_model}_{args.lr}_{args.shot}shot.json')
