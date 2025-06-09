import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.optim import Adam
import torch.autograd as autograd


class MF(nn.Module):
    def __init__(self, config):
        super(MF, self).__init__()
        self.config = config
        self.user_emb = torch.nn.Embedding(self.config["n_users"], self.config["dim"])
        self.item_emb = torch.nn.Embedding(self.config["n_items"], self.config["dim"])
        normal_(self.user_emb.weight.data, std=0.1)
        normal_(self.item_emb.weight.data, std=0.1)
        self.device = self.config["device"]
        self.f = nn.Sigmoid()

    def _hvp(self, loss, params, v):
        """
        计算 Hessian - vector product (HVP)
        :param loss: 标量损失值
        :param params: 模型参数列表
        :param v: 向量，与参数梯度同维度
        """
        # 计算一阶梯度
        grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
        # 计算 HVP
        hvp = autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        return hvp

    def forward(self, user_list, pos_item_list, neg_item_list, pos_scores=None, neg_scores=None):
        user_emb = self.user_emb(torch.LongTensor(user_list).to(self.device))
        pos_item_emb = self.item_emb(torch.LongTensor(pos_item_list).to(self.device))
        neg_item_emb = self.item_emb(torch.LongTensor(neg_item_list).to(self.device))
        reg = (user_emb.norm(dim=1).pow(2).mean() + pos_item_emb.norm(dim=1).pow(2).mean() + neg_item_emb.norm(
            dim=1).pow(2).mean())
        rating_loss = None
        if pos_scores is not None and neg_scores is not None:
            pos_scores = torch.tensor(pos_scores, dtype=torch.float32, device=self.device)
            neg_scores = torch.tensor(neg_scores, dtype=torch.float32, device=self.device)
            pos_scores_pred = torch.sum(user_emb * pos_item_emb, dim=1)
            neg_scores_pred = torch.sum(user_emb * neg_item_emb, dim=1)
            rating_loss = nn.MSELoss()(pos_scores_pred, pos_scores) + nn.MSELoss()(neg_scores_pred, neg_scores)
        return user_emb, pos_item_emb, neg_item_emb, reg, rating_loss

    def predict(self, user_list):
        # 确保user_list是torch.long类型
        user_list = torch.tensor(user_list, dtype=torch.long, device=self.device) if not isinstance(user_list,
                                                                                                    torch.Tensor) else user_list
        if user_list.dtype != torch.long:
            user_list = user_list.long()
        user_emb = self.user_emb(user_list)
        scores = torch.mm(user_emb, self.item_emb.weight.t())
        return scores

    def fit(self, train_data, train_labels):
        user_list, item_list = train_data
        optimizer = Adam(self.parameters(), lr=self.config["lr"])

        for epoch in range(self.config["epochs"]):
            self.train()
            optimizer.zero_grad()
            user_emb, item_emb, reg, rating_loss = self.forward(user_list, item_list, train_labels)
            loss = rating_loss + self.config["reg"] * reg
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{self.config['epochs']}, Loss: {loss.item()}")

    def calc_robustness_influence(self, user_list, item_list, scores, epsilon=0.1, use_hvp=True):
        # 修改这里，增加一个变量来接收多余的返回值
        user_emb, pos_item_emb, neg_item_emb, reg, rating_loss = self.forward(user_list, item_list, scores)
        user_emb = user_emb.requires_grad_(True)
        item_emb = pos_item_emb.requires_grad_(True)  # 这里应该使用 pos_item_emb

        scores_pred = torch.sum(user_emb * item_emb, dim=1)
        loss = nn.MSELoss()(scores_pred, torch.tensor(scores, dtype=torch.float32, device=self.device))

        if use_hvp:
            # 使用 HVP 计算影响值
            params = list(self.parameters())
            v = [torch.randn_like(p) for p in params]  # 随机向量
            hvp = self._hvp(loss, params, v)
            influence_values = torch.abs(hvp[0]).sum(dim=1).detach().cpu().numpy()
        else:
            # 原有一阶梯度方法
            grads = autograd.grad(loss, [user_emb], retain_graph=True)
            grad_user = grads[0]
            influence_values = torch.abs(grad_user).sum(dim=1).detach().cpu().numpy()

        return influence_values

    def calc_ferm_influence(self, user_list, item_list, scores, s, epsilon=0.1):
        user_emb, pos_item_emb, neg_item_emb, _, _ = self.forward(user_list, item_list, scores)
        user_emb = user_emb.requires_grad_(True)
        pos_item_emb = pos_item_emb.requires_grad_(True)
        idx_grp_0 = [i for i in range(len(s)) if s[i] == 0]
        idx_grp_1 = [i for i in range(len(s)) if s[i] == 1]
        user_emb_grp_0 = user_emb[idx_grp_0]
        user_emb_grp_1 = user_emb[idx_grp_1]
        item_emb_grp_0 = pos_item_emb[idx_grp_0]
        item_emb_grp_1 = pos_item_emb[idx_grp_1]
        scores_grp_0 = torch.tensor([scores[i] for i in idx_grp_0], dtype=torch.float32, device=self.device)
        scores_grp_1 = torch.tensor([scores[i] for i in idx_grp_1], dtype=torch.float32, device=self.device)
        scores_pred_grp_0 = torch.sum(user_emb_grp_0 * item_emb_grp_0, dim=1)
        scores_pred_grp_1 = torch.sum(user_emb_grp_1 * item_emb_grp_1, dim=1)
        loss_grp_0 = nn.MSELoss()(scores_pred_grp_0, scores_grp_0)
        loss_grp_1 = nn.MSELoss()(scores_pred_grp_1, scores_grp_1)
        ferm_loss = loss_grp_0 - loss_grp_1
        grads = torch.autograd.grad(ferm_loss, [user_emb], retain_graph=True)
        grad_user = grads[0]
        influence_values = torch.abs(grad_user).sum(dim=1).detach().cpu().numpy()
        return influence_values

    def calculate_influence(self, user_list, item_list, scores, s, robustness_weight, ferm_weight):
        """计算组合影响值"""
        robustness_influence = self.calc_robustness_influence(user_list, item_list, scores)
        ferm_influence = self.calc_ferm_influence(user_list, item_list, scores, s)
        # 组合影响
        combined_influence = robustness_weight * robustness_influence + ferm_weight * ferm_influence
        return combined_influence
