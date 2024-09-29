"""
General utilities
"""

from typing import List, Optional, Tuple,Dict

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyfsl.methods import FewShotClassifier


def plot_images(images: Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def predict_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict embeddings for a dataloader.
    Args:
        dataloader: dataloader to predict embeddings for. Must deliver tuples (images, class_names)
        model: model to use for prediction
        device: device to cast the images to. If none, no casting is performed. Must be the same as
            the device the model is on.
    Returns:
        dataframe with columns embedding and class_name
    """
    all_embeddings = []
    all_class_names = []
    with torch.no_grad():
        for images, class_names in tqdm(
            dataloader, unit="batch", desc="Predicting embeddings"
        ):
            if device is not None:
                images = images.to(device)
            all_embeddings.append(model(images).detach().cpu())
            if isinstance(class_names, torch.Tensor):
                all_class_names += class_names.tolist()
            else:
                all_class_names += class_names

    concatenated_embeddings = torch.cat(all_embeddings)

    return pd.DataFrame(
        {"embedding": list(concatenated_embeddings), "class_name": all_class_names}
    )


def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    number_of_correct_predictions = int(
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels)

#在原始脚本中用于计算val test阶段的acc
def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> float:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return correct_predictions / total_predictions

#这给修改是为了计算val阶段的acc和loss
LOSS_FUNCTION = nn.CrossEntropyLoss()

def evaluate_loss(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        tuple of average classification accuracy and average loss
    """
    total_predictions = 0
    correct_predictions = 0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                model.process_support_set(support_images, support_labels)
                classification_scores = model(query_images)

                loss = LOSS_FUNCTION(classification_scores, query_labels)
                total_loss += loss.item() * len(query_labels)  # Summing scaled loss
                
                correct, total = evaluate_on_one_task(
                    model,
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                )

                total_predictions += total
                correct_predictions += correct

                average_accuracy = correct_predictions / total_predictions
                average_loss = total_loss / total_predictions
                tqdm_eval.set_postfix(accuracy=average_accuracy, loss=average_loss)

    return average_accuracy, average_loss

#以下是为了计算test阶段中的各个指标
def evaluate_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[List[int], List[int]]:
    """
    Returns the predictions and true labels of query labels.
    """
    model.process_support_set(support_images, support_labels)
    predictions = torch.max(model(query_images).detach(), 1)[1].cpu().numpy()
    true_labels = query_labels.cpu().numpy()
    return predictions, true_labels

from sklearn.metrics import precision_recall_fscore_support

def test_evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on few-shot classification tasks
    Returns:
        A dictionary containing average loss, accuracy, precision, recall, and f1-score.
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    total_loss = 0.0

    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                model.process_support_set(support_images, support_labels)
                classification_scores = model(query_images)

                loss = LOSS_FUNCTION(classification_scores, query_labels)
                total_loss += loss.item() * len(query_labels)

                predictions, true_labels = evaluate_task(
                    model, support_images, support_labels, query_images, query_labels
                )
                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='macro')
    average_loss = total_loss / len(data_loader.dataset)
    accuracy = np.mean(np.array(all_true_labels) == np.array(all_predictions))
    
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

#以下是为了画test阶段的roc曲线
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_roc(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
    # save_path: Optional[str] = "./roc_curve.png"  # ROC 曲线保存路径
) -> float:
    """
    Evaluate the model on few-shot classification tasks and optionally plot the ROC curve.
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
        plot_roc: whether to plot the ROC curve
        save_path: where to save the ROC curve plot
    Returns:
        average classification accuracy
    """
    total_predictions = 0
    correct_predictions = 0

    # Store true labels and predicted probabilities for ROC curve
    all_true_labels = []
    all_predicted_probs = []

    model.eval()
    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                # Model prediction on query images
                model.process_support_set(support_images, support_labels)
                query_outputs = model(query_images)
                
                # Get the predicted probabilities
                predicted_probs = torch.softmax(query_outputs, dim=1).cpu().numpy()
                all_predicted_probs.extend(predicted_probs)

                # Get the predicted labels (argmax)
                predicted_labels = np.argmax(predicted_probs, axis=1)

                # Get the true labels
                true_labels = query_labels.cpu().numpy()
                all_true_labels.extend(true_labels)

                # Calculate correct and total predictions
                correct = np.sum(predicted_labels == true_labels)
                total = len(true_labels)

                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    accuracy = correct_predictions / total_predictions

    # Plot ROC curve if required
    
    return accuracy,all_true_labels, all_predicted_probs

#可视化
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import Tensor
from typing import Optional, Tuple

def plot_tsne(features: Tensor, labels: Tensor, title: str, filename: str):
    """
    Plot t-SNE results and save to file.
    
    Args:
        features: Tensor of features to plot.
        labels: Tensor of corresponding labels.
        title: Title of the plot.
        filename: The filename to save the plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features.cpu().numpy())
    
    plt.figure(figsize=(10, 7))
    unique_labels = torch.unique(labels).cpu().numpy()
    for label in unique_labels:
        indices = labels.cpu().numpy() == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {label}', alpha=0.6)

    plt.title(title)
    plt.legend()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

def visual_ontest(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
    visualize: bool = False,  # 添加 visualize 参数
    save_path: str = None     # 添加 save_path 参数，用于保存可视化图像
) -> None:
    """
    Perform t-SNE visualization on few-shot classification tasks.
    
    Args:
        model: a few-shot classifier.
        data_loader: loads data in the shape of few-shot classification tasks.
        device: where to cast data tensors. Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar.
        tqdm_prefix: prefix of the tqdm bar.
        visualize: whether to visualize t-SNE results.
        save_path: where to save the t-SNE plot, based on args.
        
    Returns:
        None. Only saves t-SNE visualization.
    """
    all_true_labels = []
    all_query_features = []  # 保存用于 t-SNE 可视化的特征

    # eval mode affects the behavior of certain layers (e.g., dropout, batch normalization)
    model.eval()
    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                # Perform feature extraction for t-SNE
                query_features = model.compute_features(query_images.to(device)).cpu()
                all_true_labels.extend(query_labels.cpu().numpy())
                all_query_features.append(query_features)

    # Concatenate all query features for t-SNE visualization
    all_query_features = torch.cat(all_query_features, dim=0)

    # t-SNE visualization if visualize is True
    if visualize and save_path:
        plot_tsne(all_query_features, torch.tensor(all_true_labels), title="t-SNE Visualization of Test Results", filename=save_path)
 
#可视化二





#以下是集成f1 以及 roc曲线
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    number_of_correct_predictions = int(
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels), query_labels, predictions


def evaluate_ontest(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> Tuple[float, Tensor, Tensor, dict]:
    """
    Evaluate the model on few-shot classification tasks and compute evaluation metrics.
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks.
        device: where to cast data tensors. Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar.
        tqdm_prefix: prefix of the tqdm bar.
    Returns:
        - Average classification accuracy.
        - True labels (all_true_labels).
        - Predicted probabilities (all_predicted_probs).
        - A dictionary containing precision, recall, and F1 score.
    """
    total_predictions = 0
    correct_predictions = 0
    all_true_labels = []
    all_predicted_probs = []

    # eval mode affects the behavior of certain layers (e.g., dropout, batch normalization)
    model.eval()
    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total, true_labels, predicted_probs = evaluate_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                total_predictions += total
                correct_predictions += correct

                # Append true labels and predicted probabilities for ROC and other metrics
                all_true_labels.extend(true_labels.cpu().numpy())
                all_predicted_probs.extend(predicted_probs.cpu().numpy())

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

   
    # Calculate metrics using scikit-learn
    predicted_classes = torch.argmax(torch.tensor(all_predicted_probs), dim=1).numpy()
    # precision = precision_score(all_true_labels, predicted_classes, average='weighted')
    # recall = recall_score(all_true_labels, predicted_classes, average='weighted')
    # f1 = f1_score(all_true_labels, predicted_classes, average='weighted')
    # accuracy = correct_predictions / total_predictions

    # metrics = {
    #     "precision": precision,
    #     "recall": recall,
    #     "f1_score": f1,
    #     "accuracy": accuracy
    # }
    metrics = {}
    
    try:
        precision = precision_score(all_true_labels, predicted_classes, average='weighted')
        metrics["precision"] = precision
    except ValueError as e:
        print(f"Error calculating precision: {e}")
        metrics["precision"] = None

    try:
        recall = recall_score(all_true_labels, predicted_classes, average='weighted')
        metrics["recall"] = recall
    except ValueError as e:
        print(f"Error calculating recall: {e}")
        metrics["recall"] = None

    try:
        f1 = f1_score(all_true_labels, predicted_classes, average='weighted')
        metrics["f1_score"] = f1
    except ValueError as e:
        print(f"Error calculating F1 score: {e}")
        metrics["f1_score"] = None

    # Calculate accuracy (it should not raise an exception)
    accuracy = correct_predictions / total_predictions
    metrics["accuracy"] = accuracy
    
    return accuracy, torch.tensor(all_true_labels), torch.tensor(all_predicted_probs), metrics


#以下是为了在test阶段可视化
def test_visual(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    all_predictions = []
    all_true_labels = []
    total_loss = 0.0

    with torch.no_grad():
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                model.process_support_set(support_images, support_labels)
                classification_scores = model(query_images)

                loss = LOSS_FUNCTION(classification_scores, query_labels)
                total_loss += loss.item() * len(query_labels)

                predictions, true_labels = evaluate_task(
                    model, support_images, support_labels, query_images, query_labels
                )
                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)

                # 可视化特征
                query_features = model.encode_query_features(model.compute_features(query_images))
                model.visualize_features(query_features, query_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='macro')
    average_loss = total_loss / len(data_loader.dataset)
    accuracy = np.mean(np.array(all_true_labels) == np.array(all_predictions))
    
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    

def compute_average_features_from_images(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
):
    """
    Compute the average features vector from all images in a DataLoader.
    Assumes the images are always first element of the batch.
    Returns:
        Tensor: shape (1, feature_dimension)
    """
    all_embeddings = torch.stack(
        predict_embeddings(dataloader, model, device)["embedding"].to_list()
    )
    average_features = all_embeddings.mean(dim=0)
    if device is not None:
        average_features = average_features.to(device)
    return average_features
