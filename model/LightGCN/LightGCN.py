import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.nn.functional import mse_loss
import torch.autograd as autograd

class LightGCN(nn.Module):
    def __init__(self, config, graph):
        super(LightGCN, self).__init__()
        self.config = config
        self.embedding_user = nn.Embedding(self.config["n_users"], self.config["dim"])
        self.embedding_item = nn.Embedding(self.config["n_items"], self.config["dim"])
        self.norm_adj = graph.to(self.config["device"])
        self.n_layers = self.config["n_layers"]
        self.device = self.config["device"]
        self.dropout = nn.Dropout(p=config.get("dropout_rate", 0.1))
        normal_(self.embedding_user.weight.data, std=0.1)
        normal_(self.embedding_item.weight.data, std=0.1)
        self.f = nn.Sigmoid()

    def _hvp(self, loss, params, v):
        """
        计算 Hessian-vector product (HVP)
        :param loss: 标量损失值
        :param params: 模型参数列表
        :param v: 向量，与参数梯度同维度
        """
        # 计算一阶梯度
        grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
        # 计算 HVP
        hvp = autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        return hvp

    def _get_rep(self):
        users_emb = self.dropout(self.embedding_user.weight).to(self.device)
        items_emb = self.dropout(self.embedding_item.weight).to(self.device)
        representations = torch.cat([users_emb, items_emb])
        all_layer_rep = [representations]
        for _ in range(self.n_layers):
            representations = torch.sparse.mm(self.norm_adj, representations)
            all_layer_rep.append(self.dropout(representations))
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        users, items = torch.split(final_rep, [self.config["n_users"], self.config["n_items"]])
        return users, items

    def forward(self, user_list, pos_items, neg_items, pos_scores=None, neg_scores=None):
        user_r, item_r = self._get_rep()
        user_emb = user_r[user_list]
        posI_emb = item_r[pos_items]
        negI_emb = item_r[neg_items]
        reg = (user_emb.norm(dim=1).pow(2).mean() +
               posI_emb.norm(dim=1).pow(2).mean() +
               negI_emb.norm(dim=1).pow(2).mean())
        rating_loss = None
        if pos_scores is not None and neg_scores is not None:
            pos_scores = torch.tensor(pos_scores, dtype=torch.float32, device=self.device)
            neg_scores = torch.tensor(neg_scores, dtype=torch.float32, device=self.device)
            pos_scores_pred = torch.sum(user_emb * posI_emb, dim=1)
            neg_scores_pred = torch.sum(user_emb * negI_emb, dim=1)
            rating_loss = mse_loss(pos_scores_pred, pos_scores) + mse_loss(neg_scores_pred, neg_scores)
        return user_emb, posI_emb, negI_emb, reg, rating_loss

    def predict(self, user_list):
        user_r, item_r = self._get_rep()
        user_emb = user_r[user_list]
        item_emb = item_r
        scores = torch.matmul(user_emb, item_emb.t())
        return self.f(scores)

    def calc_robustness_influence(self, user_list, pos_items, neg_items, epsilon=0.1, use_hvp=True):
        user_r, item_r = self._get_rep()
        user_emb = user_r[user_list].requires_grad_(True)
        posI_emb = item_r[pos_items].requires_grad_(True)
        negI_emb = item_r[neg_items].requires_grad_(True)

        pos_scores = torch.sum(user_emb * posI_emb, dim=1)
        neg_scores = torch.sum(user_emb * negI_emb, dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        if use_hvp:
            # 使用 HVP 计算影响值
            params = list(self.parameters())
            v = [torch.randn_like(p) for p in params]  # 随机向量
            hvp = self._hvp(loss, params, v)
            influence_values = torch.abs(hvp[0]).sum(dim=1).detach().cpu().numpy()
        else:
            # 原有一阶梯度方法
            grads = autograd.grad(loss, [user_emb, posI_emb, negI_emb], retain_graph=True)
            grad_user, grad_pos, grad_neg = grads
            influence_values = torch.abs(grad_user).sum(dim=1).detach().cpu().numpy()

        return influence_values

    def calc_ferm_influence(self, user_list, pos_items, neg_items, s, epsilon=0.1):
        user_r, item_r = self._get_rep()
        user_emb = user_r[user_list].requires_grad_(True)
        posI_emb = item_r[pos_items].requires_grad_(True)
        negI_emb = item_r[neg_items].requires_grad_(True)
        idx_grp_0 = [i for i in range(len(s)) if s[i] == 0]
        idx_grp_1 = [i for i in range(len(s)) if s[i] == 1]
        user_emb_grp_0 = user_emb[idx_grp_0]
        user_emb_grp_1 = user_emb[idx_grp_1]
        posI_emb_grp_0 = posI_emb[idx_grp_0]
        posI_emb_grp_1 = posI_emb[idx_grp_1]
        negI_emb_grp_0 = negI_emb[idx_grp_0]
        negI_emb_grp_1 = negI_emb[idx_grp_1]
        pos_scores_grp_0 = torch.sum(user_emb_grp_0 * posI_emb_grp_0, dim=1)
        pos_scores_grp_1 = torch.sum(user_emb_grp_1 * posI_emb_grp_1, dim=1)
        neg_scores_grp_0 = torch.sum(user_emb_grp_0 * negI_emb_grp_0, dim=1)
        neg_scores_grp_1 = torch.sum(user_emb_grp_1 * negI_emb_grp_1, dim=1)
        loss_grp_0 = -torch.log(torch.sigmoid(pos_scores_grp_0 - neg_scores_grp_0)).mean()
        loss_grp_1 = -torch.log(torch.sigmoid(pos_scores_grp_1 - neg_scores_grp_1)).mean()
        ferm_loss = loss_grp_0 - loss_grp_1
        grads = torch.autograd.grad(ferm_loss, [user_emb, posI_emb, negI_emb], retain_graph=True)
        grad_user, grad_pos, grad_neg = grads
        influence_values = torch.abs(grad_user).sum(dim=1).detach().cpu().numpy()
        return influence_values

    def calculate_influence(self, user_list, pos_items, neg_items, s, robustness_weight, ferm_weight):
        """计算组合影响值"""
        robustness_influence = self.calc_robustness_influence(user_list, pos_items, neg_items)
        ferm_influence = self.calc_ferm_influence(user_list, pos_items, neg_items, s)
        # 组合影响
        combined_influence = robustness_weight * robustness_influence + ferm_weight * ferm_influence
        return combined_influence
