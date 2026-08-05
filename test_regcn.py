import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=== 正在初始化 SRTP 项目：RE-GCN 完整训练与测试脚本 ===")

# 1. 数据加载器
class TKGDataset:
    def __init__(self, data_dir):
        self.entities = self.load_map(os.path.join(data_dir, "entity2id.txt"))
        self.relations = self.load_map(os.path.join(data_dir, "relation2id.txt"))
        self.train_data = self.load_quadruples(os.path.join(data_dir, "train.txt"))
        
    def load_map(self, path):
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    def load_quadruples(self, path):
        quadruples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    quadruples.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
        return np.array(quadruples)

# 2. 原生 RGCN 单元
class NativeRGCNCell(nn.Module):
    def __init__(self, num_entities, num_relations, h_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.h_dim = h_dim
        self.w_relation = nn.Parameter(torch.Tensor(num_relations * 2, h_dim, h_dim))
        nn.init.xavier_uniform_(self.w_relation)

    def forward(self, h_prev, quadruplets):
        if len(quadruplets) == 0:
            return h_prev
        s, r, o = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2]
        sub_emb = h_prev[s]
        rel_w = self.w_relation[r]
        msg = torch.bmm(sub_emb.unsqueeze(1), rel_w).squeeze(1)
        
        h_out = torch.zeros_like(h_prev)
        h_out.index_add_(0, torch.tensor(o, device=h_prev.device), msg)
        return F.relu(h_out)

# 3. 完整的 RE-GCN 模型（含评分函数用于预测）
class REGCNCompleteModel(nn.Module):
    def __init__(self, num_entities, num_relations, h_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.h_dim = h_dim
        
        self.ent_embeds = nn.Parameter(torch.Tensor(num_entities, h_dim))
        self.rel_embeds = nn.Parameter(torch.Tensor(num_relations, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds)
        nn.init.xavier_uniform_(self.rel_embeds)
        
        self.rgcn = NativeRGCNCell(num_entities, num_relations, h_dim)
        self.gru = nn.GRUCell(h_dim, h_dim)
        
    def get_entity_representations(self, quadruplets):
        timestamps = np.unique(quadruplets[:, 3])
        h_t = self.ent_embeds
        for t in timestamps:
            snap_mask = (quadruplets[:, 3] == t)
            snap_triplets = quadruplets[snap_mask][:, :3]
            gcn_out = self.rgcn(h_t, snap_triplets)
            h_t = self.gru(gcn_out, h_t)
        return h_t  # 返回演化后的实体时序表征 [num_entities, h_dim]

    def score_func(self, h, sub, rel, obj):
        # 简单的 DistMult 评分函数: <s, r, o>
        s_emb = h[sub]
        r_emb = self.rel_embeds[rel]
        o_emb = h[obj]
        score = torch.sum(s_emb * r_emb * o_emb, dim=-1)
        return score

    def forward(self, quadruplets):
        # 1. 获得时序演化后的实体表示
        h = self.get_entity_representations(quadruplets)
        
        # 2. 针对当前批次计算正样本得分
        sub, rel, obj = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2]
        pos_scores = self.score_func(h, sub, rel, obj)
        
        # 3. 简单负采样：随机替换客体 obj 构造负样本
        neg_obj = torch.randint(0, self.num_entities, obj.shape, device=obj.device)
        neg_scores = self.score_func(h, sub, rel, neg_obj)
        
        # 4. 使用 Margin Ranking Loss（边界排序损失）
        criterion = nn.MarginRankingLoss(margin=1.0)
        target = torch.ones_like(pos_scores)
        loss = criterion(pos_scores, neg_scores, target)
        
        return loss

# 4. 训练主流程
if __name__ == "__main__":
    data_dir = "./data/sample_tkg"
    dataset = TKGDataset(data_dir)
    
    num_ent = len(dataset.entities)
    num_rel = len(dataset.relations)
    h_dim = 32
    
    model = REGCNCompleteModel(num_ent, num_rel, h_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_tensor = torch.tensor(dataset.train_data, dtype=torch.long)
    
    print("\n🚀 开始训练 RE-GCN 模型...")
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        loss = model(train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
            
    print("\n✅ RE-GCN 模型训练完成！")
    
    # 模型评估/预测演示
    model.eval()
    with torch.no_grad():
        final_h = model.get_entity_representations(train_tensor)
        # 测试预测：“刘备(0) 结义(0) 关羽(1)” 的合理性得分
        test_score = model.score_func(final_h, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        print(f"🔗 事实 (刘备 -> 结义 -> 关羽) 关联得分: {test_score.item():.4f}")
