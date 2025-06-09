import json
import pickle
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.patches as mpatches
from model.LightGCN.LightGCN import LightGCN
from model.MF.MF import MF


class BasicDataset(Dataset):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.config = config
        self.filter_stage = "initial"  # 可以是 "initial", "filtering", "final"
        self.load_data()

    def set_filter_stage(self, stage):
        """设置当前过滤阶段"""
        assert stage in ["initial", "filtering", "final"]
        self.filter_stage = stage

    def load_data(self):
        # 根据当前阶段决定是否加载过滤后的数据
        if self.filter_stage == "initial":
            # 加载原始数据
            self._load_initial_data()
        elif self.filter_stage == "filtering" and self.config["with_impact_filtering"]:
            # 计算影响值并保存过滤数据
            self._calculate_and_filter()
        else:  # final
            if self.config["with_impact_filtering"]:
                # 加载过滤后的数据
                self._load_filtered_data()
            else:
                # 如果不进行过滤，直接使用原始数据
                self._load_initial_data()

    def get_interactions_with_influence(self):
        influence_values = self._calculate_influence_values()
        user_interactions = self.user_interactions
        all_user_ids = list(user_interactions.keys())
        all_item_ids = []
        all_scores = []
        all_influence = []
        for user in all_user_ids:
            interactions = user_interactions[user]
            for item, score in interactions:
                all_item_ids.append(item)
                all_scores.append(score)
                if user in influence_values and item in influence_values[user]:
                    all_influence.append(influence_values[user][item])
                else:
                    all_influence.append(0)
        return all_item_ids, all_scores, all_influence

    def _load_initial_data(self):
        data_path = os.path.join(self.path, "data.txt")
        if self.config["dataset"] == "ML-100k":
            data = pd.read_csv(data_path, sep="\t", header=None, names=["UserID", "ItemID", "Score", "Timestamp"])
        else:
            data = pd.read_csv(data_path, sep="::", header=None, names=["UserID", "ItemID", "Score", "Timestamp"])
        # Filter users and items with fewer interactions than min_interaction
        user_interactions = data["UserID"].value_counts()
        item_interactions = data["ItemID"].value_counts()
        filtered_users = user_interactions[user_interactions >= self.config["min_interaction"]].index
        filtered_items = item_interactions[item_interactions >= self.config["min_interaction"]].index
        data = data[data["UserID"].isin(filtered_users) & data["ItemID"].isin(filtered_items)]
        # Normalize scores to [0, 1]
        min_score, max_score = data["Score"].min(), data["Score"].max()
        data["Score"] = (data["Score"] - min_score) / (max_score - min_score)
        # Create user and item mappings
        user_ids = {user: idx for idx, user in enumerate(filtered_users)}
        item_ids = {item: idx for idx, item in enumerate(filtered_items)}
        # Create user-item interaction dictionary
        user_item_dict = defaultdict(list)
        for _, row in data.iterrows():
            user_id = user_ids[row["UserID"]]
            item_id = item_ids[row["ItemID"]]
            score = row["Score"]
            user_item_dict[user_id].append((item_id, score))
        # Save processed data
        if not os.path.exists(os.path.join(self.path, "processed")):
            os.makedirs(os.path.join(self.path, "processed"), exist_ok=True)
        with open(os.path.join(self.path, f"processed/user_interactions_more_{self.config['min_interaction']}.pickle"),
                  'wb') as f:
            pickle.dump((user_item_dict, len(user_ids), len(item_ids), item_ids), f)
        self.user_interactions, self.n_users, self.n_items, self.item_map = self._load_org_data()
        if self.config["with_fakes"]:
            self._load_fakes(self.item_map)

    def _calculate_and_filter(self):
        """计算影响值并过滤数据"""
        if not hasattr(self, 'model'):
            raise ValueError("Model must be set before filtering")

        print("Calculating influence values for filtering...")
        influence_values = self._calculate_influence_values()

        # 统计每个物品的交互次数
        item_interaction_count = defaultdict(int)
        for user_interactions in self.user_interactions.values():
            for item, _ in user_interactions:
                item_interaction_count[item] += 1

        # 筛选冷启动物品（交互次数少于10）
        item_ids_cold = []
        user_ids_cold = []
        item_ids_normal = []
        user_ids_normal = []
        for user, interactions in self.user_interactions.items():
            for item, _ in interactions:
                if item_interaction_count[item] < 2:
                    item_ids_cold.append(item)
                    user_ids_cold.append(user)
                else:
                    item_ids_normal.append(item)
                    user_ids_normal.append(user)


    def _calculate_influence_values(self):
        influence_values = {}
        for user in range(self.n_users):
            interactions = self.user_interactions[user]
            if not interactions:
                continue

            # 计算每个用户 - 物品对的影响值
            for item, score in interactions:
                # 随机生成负样本
                neg_item = self._get_negative_item(user)

                # 计算组合影响值
                influence = self.model.calculate_influence(
                    [user], [item], [neg_item],
                    [len(interactions)],  # 使用交互次数作为敏感属性
                    self.config["robustness_weight"],
                    self.config["ferm_weight"]
                )
                if user not in influence_values:
                    influence_values[user] = {}
                influence_values[user][item] = influence[0]
        return influence_values

    def _filter_by_influence(self, influence_values, keep_ratio):
        filtered_interactions = {}
        for user in influence_values:
            user_interactions = self.user_interactions[user]
            # 将交互与影响值配对
            inter_with_influence = [
                (item, score, influence_values[user][item])
                for item, score in user_interactions
                if item in influence_values[user]
            ]
            # 按影响值降序排序
            inter_with_influence.sort(key=lambda x: -x[2])
            # 计算保留数量
            keep_num = max(1, int(len(inter_with_influence) * keep_ratio))
            # 保留高影响交互
            filtered_interactions[user] = [
                (item, score) for item, score, _ in inter_with_influence[:keep_num]
            ]
        return filtered_interactions

    def _apply_filter(self, influence_values):
        """应用过滤"""
        keep_ratio = self.config.get("keep_ratio", 0.8)
        self.user_interactions = self._filter_by_influence(influence_values, keep_ratio)

    def _save_filtered_data(self):
        """保存过滤后的数据"""
        impact_suffix = "_impact_filtered" if self.config["with_impact_filtering"] else ""
        pickle_file_name = f"processed/user_interactions_more_{self.config['min_interaction']}_impact_filtered_keep_{self.config['keep_ratio']}_robust_{self.config['robustness_weight']}_ferm_{self.config['ferm_weight']}.pickle"
        pickle_file_path = os.path.join(self.path, pickle_file_name)
        with open(pickle_file_path, 'wb') as f:
            pickle.dump((self.user_interactions, self.n_users, self.n_items), f)

    def _load_filtered_data(self):
        """加载过滤后的数据"""
        impact_suffix = "_impact_filtered" if self.config["with_impact_filtering"] else ""
        pickle_file_name = f"processed/user_interactions_more_{self.config['min_interaction']}{impact_suffix}_{self.config['keep_ratio']}+{self.config['robustness_weight']}.pickle"
        pickle_file_path = os.path.join(self.path, pickle_file_name)
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                self.user_interactions, self.n_users, self.n_items = pickle.load(f)
        else:
            raise FileNotFoundError(f"Filtered data not found at {pickle_file_path}")

    def _load_org_data(self):
        data_path = os.path.join(self.path, "data.txt")
        if self.config["dataset"] == "ML-100k":
            data = pd.read_csv(data_path, sep="\t", header=None, names=["UserID", "ItemID", "Score", "Timestamp"])
        else:
            data = pd.read_csv(data_path, sep="::", header=None, names=["UserID", "ItemID", "Score", "Timestamp"])
        user_interactions = data["UserID"].value_counts()
        item_interactions = data["ItemID"].value_counts()
        filtered_users = user_interactions[user_interactions >= self.config["min_interaction"]].index
        filtered_items = item_interactions[item_interactions >= self.config["min_interaction"]].index
        data = data[data["UserID"].isin(filtered_users) & data["ItemID"].isin(filtered_items)]
        min_score, max_score = data["Score"].min(), data["Score"].max()
        data["Score"] = (data["Score"] - min_score) / (max_score - min_score)
        user_ids = {user: idx for idx, user in enumerate(filtered_users)}
        item_ids = {item: idx for idx, item in enumerate(filtered_items)}
        user_item_dict = defaultdict(list)
        for _, row in data.iterrows():
            user_id = user_ids[row["UserID"]]
            item_id = item_ids[row["ItemID"]]
            score = row["Score"]
            user_item_dict[user_id].append((item_id, score))
        return user_item_dict, len(user_ids), len(item_ids), item_ids

    def _load_fakes(self, item_map):
        with open(os.path.join(self.path, f"{self.config['attack_type']}.json"), 'r') as f:
            fake_data = json.load(f)
        fake_user_interactions, target_items = fake_data["fake_users"], fake_data["target_items"]
        self.target_item = [item_map[item] for item in target_items]
        self.n_fake_users = 0
        for _, fake_interaction in fake_user_interactions.items():
            if len(fake_interaction) < self.config["min_interaction"]:
                continue
            interactions = []
            for item, score in fake_interaction:
                if item in item_map:
                    interactions.append((item_map[item], score))
            if interactions:
                self.user_interactions[self.n_users + self.n_fake_users] = interactions
                self.n_fake_users += 1
        self.n_users += self.n_fake_users

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return index

    def _get_negative_item(self, user):
        all_interacted = set([item for item, _ in self.user_interactions[user]])
        while True:
            neg_item = random.randint(0, self.n_items - 1)
            if neg_item not in all_interacted:
                break
        return neg_item

    def get_train_batch(self, inter_list):
        inter_list = inter_list.squeeze().tolist()
        pos_item_list = []
        neg_item_list = []
        pos_scores_list = []
        neg_scores_list = []
        user_list = np.random.randint(0, self.n_users, len(inter_list))
        for user in user_list:
            all_interacted = set([item for item, _ in self.user_interactions[user]])
            pos_item, pos_score = random.choice(self.user_interactions[user])
            pos_item_list.append(pos_item)
            pos_scores_list.append(pos_score)
            while True:
                neg_item = random.randint(0, self.n_items - 1)
                if neg_item not in all_interacted:
                    break
            neg_item_list.append(neg_item)
            neg_scores_list.append(1.0)
        return user_list, np.array(pos_item_list), np.array(neg_item_list), np.array(pos_scores_list), np.array(neg_scores_list)

    def get_val_batch(self, user_list):
        user_train_list = [self.train_data[user] for user in user_list]
        return np.array(user_list), [self.val_data[user] for user in user_list], user_train_list

    def get_test_batch(self, user_list, is_clean=False):
        if is_clean and self.config["with_fakes"]:
            new_user_list = [user for user in user_list if user < self.n_users - self.n_fake_users]
            user_list = new_user_list
        user_train_list = [self.train_data[user] for user in user_list]
        return np.array(user_list), [self.test_data[user] for user in user_list], user_train_list

    def gcn_graph(self):
        user_list = []
        item_list = []
        for user in range(self.n_users):
            items = [item for item, score in self.user_interactions[user]]
            for item in items:
                user_list.append(user)
                item_list.append(item)
        user_dim = torch.LongTensor(user_list)
        item_dim = torch.LongTensor(item_list)
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users + self.n_items, self.n_users + self.n_items]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users + self.n_items, self.n_users + self.n_items]))
        Graph = Graph.coalesce()
        return Graph


class CFDataset(BasicDataset):
    def __init__(self, path, config):
        super().__init__(path, config)
        self.split_ratio = [0.8, 0.1, 0.1]  # 8:1:1 split
        self._build_set()  # 构建训练/验证/测试集
        self._precompute_full_interactions()  # 预计算完整交互集合

    def __len__(self):
        return self.n_train_num

    def get_target_item(self):
        if self.config['attacktype']=="random":
            """获取目标物品列表，根据当前数据集配置"""
            if self.config["dataset"] == "ML-100k":
                return [1309,228,51,1518,563]
            elif self.config["dataset"] == "Filmtrust":
                return [456,102,1126,1003,914]  # ML-1M目标物品ID为1000和1001
            elif self.config["dataset"] == "Douban":
                return [7296,1639,18024,16049,14628]
            else:
                return [1824,409,4506,4012,3657]
        elif self.config['attacktype']=="unpopular":
            """获取目标物品列表，根据当前数据集配置"""
            if self.config["dataset"] == "ML-100k":
                return [1389,1493,1673,1610,1538]  # ML-100k目标物品ID为0
            elif self.config["dataset"] == "Filmtrust":
                return [466,827,1780,614,1644]  # ML-1M目标物品ID为1000和1001
            elif self.config["dataset"] == "Douban":
                return [25303,27692,20938,18306,37461]
            else:
                return [9442,9073,9522,9667,8470]
        return None

    def _build_set(self):
        """划分训练/验证/测试集"""
        self.n_train_num = 0
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]
        all_num = 0
        for user in range(self.n_users):
            interactions = self.user_interactions[user]
            n_inter_items = len(interactions)
            n_train_items = int(n_inter_items * self.split_ratio[0])
            n_val_items = int(n_inter_items * self.split_ratio[1])
            self.train_data[user] = interactions[:n_train_items]
            self.val_data[user] = interactions[n_train_items:n_train_items + n_val_items]
            self.test_data[user] = interactions[n_train_items + n_val_items:]
            self.n_train_num += n_train_items
            all_num += n_inter_items
        self.avg_inter = int(self.n_train_num / self.n_users)
        print(
            f"#User: {self.n_users}, #Item: {self.n_items}, #Ratings: {all_num}, AvgLen: {int(10 * (all_num / self.n_users)) / 10}, Sparsity: {100 - int(10000 * all_num / (self.n_users * self.n_items)) / 100}")

    def _precompute_full_interactions(self):
        """预计算每个用户的完整交互集合（用于负样本生成）"""
        self.full_interactions = []
        for user in range(self.n_users):
            self.full_interactions.append(set([item for item, _ in self.user_interactions[user]]))

    def get_val_batch(self, user_list):
        user_train_list = [self.train_data[user] for user in user_list]
        return np.array(user_list), [self.val_data[user] for user in user_list], user_train_list

    def get_test_batch(self, user_list, is_clean=False):
        if is_clean and self.config["with_fakes"]:
            new_user_list = [user for user in user_list if user < self.n_users - self.n_fake_users]
            user_list = new_user_list
        user_train_list = [self.train_data[user] for user in user_list]
        return np.array(user_list), [self.test_data[user] for user in user_list], user_train_list