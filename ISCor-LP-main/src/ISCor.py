import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn import metrics
from typing import Any, Optional, Tuple
import random
import numpy as np

class GCN(nn.Module):
    def __init__(self, feature_dims, out_dims, hidden_dims):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_dims, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dims, out_dims)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class StableTemperatureSoftmaxLayer(nn.Module):
    def __init__(self, input_dim, temperature=2.0):
        super(StableTemperatureSoftmaxLayer, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=2)
        self.temperature = temperature

    def forward(self, x):
        logits = self.linear(x)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits / self.temperature
        return F.softmax(logits, dim=1)

class Model_Net(nn.Module):
    def __init__(self, dim, layer_nums, node_nums, gcn_data, target_id, device):
        super(Model_Net, self).__init__()

        self.node_dim = dim
        self.edge_dim = dim * 2
        self.layer_nums = layer_nums
        self.node_nums = node_nums
        self.gcn_data = gcn_data
        self.target_id = target_id
        self.device = device

        for i in range(self.layer_nums):
            gcn = GCN(feature_dims=1, out_dims=self.node_dim, hidden_dims=64)
            setattr(self, 'gcn%i' % i, gcn)

        self.g_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.edge_dim))

        self.weight_softmax = StableTemperatureSoftmaxLayer(input_dim=self.node_dim + self.node_dim + self.edge_dim, temperature=2.0)

        self.d_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=self.layer_nums))

        self.p_mlp = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=2))

        self.node_ID_embedding = nn.Embedding(self.node_nums, self.node_dim)

    def forward(self, leftnode, rightnode, dis_input, indices, flag='train'):

        for i in range(self.layer_nums):
            layer_embed = eval('self.gcn' + str(i))(self.gcn_data[i]).cuda()
            setattr(self, 'layer%i' % i, layer_embed)

        TargetLayer_NodeEmbed = eval('self.layer'+str(self.target_id))
        specific_embed = torch.cat((TargetLayer_NodeEmbed[leftnode], TargetLayer_NodeEmbed[rightnode]), dim=1)

        if flag == 'train':
            layer_names = ['self.layer' + str(i) for i in dis_input[:, 2].cpu().numpy().tolist()]
            dis_embed = torch.Tensor().cuda()
            for (l, i, j) in zip(layer_names, dis_input[:, 0], dis_input[:, 1]):
                temp = torch.cat((eval(l)[i], eval(l)[j]), dim=0).cuda()
                temp = torch.unsqueeze(temp, dim=0)
                dis_embed = torch.cat((dis_embed, temp), dim=0)

            dis_embed = self.g_mlp(dis_embed)
            dis_reverse = grad_reverse(dis_embed, coeff=1)
            discriminant_out = self.d_mlp(dis_reverse)

            common_embed = dis_embed[indices]
        else:  # flag=='eval'
            common_embed = self.g_mlp(specific_embed)
            discriminant_out = -1

        weight_tensor = torch.cat((self.node_ID_embedding(leftnode), self.node_ID_embedding(rightnode), specific_embed), dim=1)
        w_out = self.weight_softmax(weight_tensor)
        w0 = w_out[:, 0].unsqueeze(1)
        w1 = w_out[:, 1].unsqueeze(1)
        p_input = torch.add(specific_embed * w0, common_embed * w1)
        prediction_out = self.p_mlp(p_input)
        return prediction_out, discriminant_out

    def loss(self, data):
        left_node = data[:, 0].to(self.device)
        right_node = data[:, 1].to(self.device)
        pred_label = data[:, 2].to(self.device)

        all_layers_list = [i for i in range(self.layer_nums)]
        dis_label = []
        for _ in range(len(pred_label)):
            random.shuffle(all_layers_list)
            dis_label.extend(all_layers_list)
        indices = torch.LongTensor(np.where(np.array(dis_label) == self.target_id)[0]).to(self.device)
        dis_label = torch.LongTensor(dis_label).to(self.device)
        dis_input = np.repeat(data, self.layer_nums, axis=0)
        dis_input[:, 2] = dis_label
        dis_input = torch.LongTensor(dis_input).to(self.device)

        pred_out, dis_out = self.forward(left_node, right_node, dis_input, indices)
        criterion = nn.CrossEntropyLoss()
        p_loss = criterion(pred_out, pred_label)
        d_loss = criterion(dis_out, dis_input[:, 2])
        whole_loss = p_loss + d_loss
        return whole_loss

    def metrics_eval(self, eval_data):
        scores = []
        labels = []
        preds = []
        for data in eval_data:
            data = data[0]
            left_node = data[:, 0].to(self.device)
            right_node = data[:, 1].to(self.device)
            link_label = data[:, 2].to(self.device)

            output, _ = self.forward(left_node, right_node, -1, -1, flag='eval')
            output = F.softmax(output, dim=1)
            _, argmax = torch.max(output, 1)
            scores += list(output[:, 1].cpu().detach().numpy())
            labels += list(link_label.cpu().detach().numpy())
            preds += list(argmax.cpu().detach().numpy())

        acc = metrics.accuracy_score(labels, preds)
        pre = metrics.precision_score(labels, preds, average='weighted')
        f1 = metrics.f1_score(labels, preds, average='weighted')
        auc = metrics.roc_auc_score(labels, scores, average=None)

        return acc, pre, f1, auc
