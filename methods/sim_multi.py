from torchvision import transforms
from torch import nn, optim
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

import torch.nn.functional as F
from .few_shot_classifier import FewShotClassifier

class SimpleShot(FewShotClassifier):
    def __init__(self,backbone:nn.Module,node_indices: list):
        super(SimpleShot, self).__init__()
        self.backbone=backbone
        self.node_indices = node_indices


    def calculate(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor, node_idx: int):

        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        train_nodes, eval_nodes = get_graph_node_names(self.backbone)
        
        return_nodes2 ={train_nodes[node_idx]: "feature"}

        f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes2)
        
        out2 = f2(support_images)

        
        out2_pooled = global_avg_pool(out2['feature'])

        z_support = out2_pooled.view(out2_pooled.size(0), -1)

        out3 = f2(query_images)
       
        out3_pooled = global_avg_pool(out3['feature'])

        z_query  = out3_pooled.view(out3_pooled.size(0), -1)
        n_way=len(torch.unique(support_labels))

        
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        dists = torch.cdist(z_query, z_proto)  #欧氏距离
        # Normalize features to compute cosine similarity
        z_query_norm = F.normalize(z_query, p=2, dim=1)
        z_proto_norm = F.normalize(z_proto, p=2, dim=1)

        # Compute cosine similarity and convert to cosine distance
        cosine_similarity = torch.mm(z_query_norm, z_proto_norm.transpose(0, 1))
        dists = 1 - cosine_similarity    
        return dists
    
    def forward(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor):
        

        distances = [self.calculate(support_images, support_labels, query_images, idx) for idx in self.node_indices]
        scores4 = -sum(distances)
        return scores4