from torchvision import transforms
from torch import nn, optim
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .few_shot_classifier import FewShotClassifier



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,gamma):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = gamma

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

    
class Simplechange(FewShotClassifier):
    def __init__(self,backbone:nn.Module,node_indices:list,weightlearnable,weight):
        super(Simplechange, self).__init__()
        self.backbone=backbone
        self.node_indices = node_indices
        self.weightlearnable=weightlearnable
        self.weight=weight
        
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor([w])) for w in weight])

        self.gamma1  =   nn.Parameter(torch.Tensor([0.0]))
        self.gamma2  =   nn.Parameter(torch.Tensor([0.0]))

    # def calculate(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor,num,numker):
    # def calculate(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor,node_idx: int):


                
    #     global_avg_pool = nn.AdaptiveAvgPool2d((1))
    #     train_nodes, eval_nodes = get_graph_node_names(self.backbone)
    #     # return_nodes2 = {
    #     #     str(train_nodes[num]): "maxpool1"
    #     # }
    #     return_nodes2 ={train_nodes[node_idx]: "feature"}
    #     f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes2)
    #     out2 = f2(support_images)
        
    #     input_size = out2['feature']  # Batch size, number of channels, sequence length
    #     n_channels = input_size.size()[1]
    #     # self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma1).to('cuda')
    #     self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma1)

    #     output,_ = self_attention(input_size)
    #     # print(output[0][0])
        
    #     out2_pooled = global_avg_pool(output)              
    #     z_support  = out2_pooled.view(out2_pooled.size(0), -1)

    #     out3 = f2(query_images)
    #     input_size = out3['feature']  # Batch size, number of channels, sequence length
    #     n_channels = input_size.size()[1]
    #     # self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma2).to('cuda')
    #     self_attention = Self_Attn(in_dim=n_channels,activation='relu',gamma=self.gamma2)

    #     output,_ = self_attention(input_size)
    #     # print(output[0][0])
        
    #     out3_pooled = global_avg_pool(output)
    #     z_query  = out3_pooled.view(out3_pooled.size(0), -1)
        
    #     n_way=len(torch.unique(support_labels))

    #     z_proto = torch.cat(
    #         [
    #             z_support[torch.nonzero(support_labels == label)].mean(0)
    #             for label in range(n_way)
    #         ]
    #     )
    #     dists = torch.cdist(z_query, z_proto)
    #     return dists
    def calculate(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor, node_idx: int):
        global_avg_pool = nn.AdaptiveAvgPool2d((1))
        train_nodes, _ = get_graph_node_names(self.backbone)
        return_nodes = {train_nodes[node_idx]: "feature"}
        f2 = create_feature_extractor(self.backbone, return_nodes=return_nodes)
        out2 = f2(support_images)
        
        input_size = out2['feature']
        self_attention = Self_Attn(in_dim=input_size.size(1), activation='relu', gamma=self.gamma1)
        output, _ = self_attention(input_size)
        out2_pooled = global_avg_pool(output)
        z_support = out2_pooled.view(out2_pooled.size(0), -1)

        out3 = f2(query_images)
        input_size = out3['feature']
        self_attention = Self_Attn(in_dim=input_size.size(1), activation='relu', gamma=self.gamma2)
        output, _ = self_attention(input_size)
        out3_pooled = global_avg_pool(output)
        z_query = out3_pooled.view(out3_pooled.size(0), -1)
        
        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [z_support[torch.nonzero(support_labels == label, as_tuple=True)].mean(0).unsqueeze(0)
            for label in range(n_way)]
        )

        # Ensure tensors are 2D
        z_query_norm = F.normalize(z_query, p=2, dim=1)
        z_proto_norm = F.normalize(z_proto, p=2, dim=1)
        
        # Ensure z_proto_norm is always a matrix, even for n_way = 1
        if z_proto_norm.dim() == 1:
            z_proto_norm = z_proto_norm.unsqueeze(0)
        
        # Compute cosine similarity
        cosine_similarity = torch.mm(z_query_norm, z_proto_norm.transpose(0, 1))
        
        # Convert cosine similarity to cosine distance
        cosine_distance = 1 - cosine_similarity

        return cosine_distance
    # def process_support_set(
    #     self,
    #     support_images: torch.Tensor,
    #     support_labels: torch.Tensor,
    # ):
    #     support_features = self.compute_features(support_images)
    #     self._validate_features_shape(support_features)
    #     self.contextualized_support_features = self.encode_support_features(
    #         support_features
    #     )

    #     self.one_hot_support_labels = (
    #         nn.functional.one_hot(  # pylint: disable=not-callable
    #             support_labels
    #         ).float()
    #     )

    def forward(self,support_images:torch.Tensor,support_labels:torch.Tensor,query_images: torch.Tensor):
        
        distances = [self.calculate(support_images, support_labels, query_images, idx) for idx in self.node_indices]
 
        if self.weightlearnable:
            scores4 = -sum(self.weights[i] * distances[i] for i in range(len(distances)))
        else:
            scores4 = -sum(self.weight[i] * distances[i] for i in range(len(distances)))
        return scores4
    
    

