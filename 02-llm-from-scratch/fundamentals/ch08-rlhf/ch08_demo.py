"""
第8章：RLHF 代码演示

这个文件演示 RLHF 的核心概念：
1. 奖励模型训练（模拟）
2. KL 散度计算
3. PPO 训练流程（模拟）

注意：这是一个简化版演示，用于理解概念。
真实的 RLHF 需要大量数据和计算资源。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRewardModel(nn.Module):
    """
    简化版奖励模型

    输入：prompt + response 的 token embeddings
    输出：标量奖励分数
    """

    def __init__(self, embed_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, embed_dim)
        x = F.relu(self.fc1(x))
        reward = self.fc2(x)
        return reward.squeeze(-1)


def train_reward_model_demo():
    """
    演示奖励模型训练过程
    """
    print("=" * 60)
    print("奖励模型训练演示")
    print("=" * 60)

    # 模拟数据：3 个 prompt-response 对
    # 每个样本有 (prompt, winner_response, loser_response)

    embed_dim = 768
    rm = SimpleRewardModel(embed_dim)
    optimizer = torch.optim.Adam(rm.parameters(), lr=1e-4)

    print("\n模拟比较数据：")
    print("Prompt: '如何保持健康？'")
    print("  回答A (好): '均衡饮食和规律运动是保持健康的关键...'")
    print("  回答B (差): '我不知道'")
    print()

    for step in range(5):
        # 模拟嵌入向量（实际中来自 LLM）
        prompt_embed = torch.randn(1, embed_dim)
        winner_embed = torch.randn(1, embed_dim)
        loser_embed = torch.randn(1, embed_dim)

        # 拼接 prompt + response
        winner_input = prompt_embed + winner_embed
        loser_input = prompt_embed + loser_embed

        # 奖励模型打分
        score_winner = rm(winner_input)
        score_loser = rm(loser_input)

        # Bradley-Terry 损失
        loss = -F.logsigmoid(score_winner - score_loser)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}:")
        print(f"  好回答分数: {score_winner.item():.3f}")
        print(f"  差回答分数: {score_loser.item():.3f}")
        print(f"  损失: {loss.item():.3f}")

    print("\n训练完成！奖励模型学会了给好回答更高的分数。")


def kl_divergence_demo():
    """
    演示 KL 散度计算
    """
    print("\n" + "=" * 60)
    print("KL 散度演示")
    print("=" * 60)

    print("\nKL 散度衡量两个概率分布的差异：")
    print("KL(P || Q) = Σ P(x) * log(P(x) / Q(x))")
    print()

    # 模拟两个 token 分布
    vocab_size = 10

    # 原始模型的分布
    p = torch.tensor([0.1, 0.2, 0.15, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05])

    # 当前模型的分布（稍微偏离）
    q_close = torch.tensor([0.12, 0.18, 0.14, 0.11, 0.06, 0.09, 0.11, 0.04, 0.09, 0.06])

    # 当前模型的分布（偏离很远）
    q_far = torch.tensor([0.3, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05, 0.15, 0.05, 0.05])

    # 计算 KL 散度
    kl_close = F.kl_div(torch.log(q_close), p, reduction="sum")
    kl_far = F.kl_div(torch.log(q_far), p, reduction="sum")

    print("原始模型分布 P: [0.10, 0.20, 0.15, 0.10, 0.05, ...]")
    print()
    print("情况1：当前模型接近原始模型")
    print(f"  分布 Q: [0.12, 0.18, 0.14, 0.11, 0.06, ...]")
    print(f"  KL 散度: {kl_close.item():.4f}")
    print()
    print("情况2：当前模型偏离原始模型")
    print(f"  分布 Q: [0.30, 0.05, 0.10, 0.05, 0.10, ...]")
    print(f"  KL 散度: {kl_far.item():.4f}")
    print()
    print("结论：偏离越远，KL 散度越大，惩罚越大！")


def ppo_training_demo():
    """
    演示 PPO 训练流程
    """
    print("\n" + "=" * 60)
    print("PPO 训练流程演示")
    print("=" * 60)

    print("\nPPO 训练步骤：")
    print("1. 从数据集采样 prompt")
    print("2. 用当前 LLM 生成 response")
    print("3. 用奖励模型计算 reward")
    print("4. 计算 KL 散度惩罚")
    print("5. 更新 LLM 参数")
    print()

    # 模拟训练过程
    embed_dim = 768
    rm = SimpleRewardModel(embed_dim)

    # 模拟参数
    beta = 0.1  # KL 惩罚系数

    print(f"KL 惩罚系数 β = {beta}")
    print()

    for step in range(3):
        # 模拟 prompt
        prompt_embed = torch.randn(1, embed_dim)

        # 模拟 LLM 生成的 response
        response_embed = torch.randn(1, embed_dim)

        # 奖励模型分数
        reward = rm(prompt_embed + response_embed)

        # 模拟 KL 散度（实际中需要计算两个分布的差异）
        kl_div = torch.rand(1) * 2  # 0 到 2 之间

        # 总奖励 = 奖励 - β * KL惩罚
        total_reward = reward - beta * kl_div

        print(f"Step {step + 1}:")
        print(f"  奖励模型分数: {reward.item():.3f}")
        print(f"  KL 散度: {kl_div.item():.3f}")
        print(f"  总奖励 = {reward.item():.3f} - {beta} * {kl_div.item():.3f}")
        print(f"        = {total_reward.item():.3f}")
        print()

    print("关键：总奖励 = 奖励 - β * KL散度")
    print("      这确保模型不会为了追求高奖励而偏离太远！")


def show_rlhf_pipeline():
    print("\n" + "=" * 60)
    print("完整 RLHF 流程总结")
    print("=" * 60)
    print("""
+-----------------------------------------------------------+
|  Stage 1: Pretraining                                  |
|  ---------------------------------------------------------- |
|  Data: Internet text (trillions of tokens)              |
|  Goal: Predict next token                                |
|  Time: Weeks/months                                     |
|  Result: Base language model (e.g., GPT-3)             |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|  Stage 2: SFT (Supervised Finetuning)                 |
|  ---------------------------------------------------------- |
|  Data: (prompt, response) pairs (10K-100K)             |
|  Goal: Learn to converse                                |
|  Time: Hours/days                                       |
|  Result: Conversational model                           |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|  Stage 3: RLHF                                           |
|  ---------------------------------------------------------- |
|  3.1 Train Reward Model                                |
|      Data: Human comparisons (prompt, winner, loser) |
|      Goal: Learn human preferences                       |
|                                                          |
|  3.2 PPO Training                                          |
|      Reward: RM score - beta * KL penalty              |
|      Goal: Generate high-reward, low-divergence responses |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
|  Final Model: ChatGPT / Claude / etc.               |
|  ---------------------------------------------------------- |
|  [OK] Can converse                                       |
|  [OK] Aligns with human preferences                 |
|  [OK] More helpful and harmless                     |
+-----------------------------------------------------------+
    """)


def main():
    print("\n" + "=" * 60)
    print("第8章：RLHF 代码演示")
    print("=" * 60)

    # 1. 奖励模型训练
    train_reward_model_demo()

    # 2. KL 散度演示
    kl_divergence_demo()

    # 3. PPO 训练流程
    ppo_training_demo()

    # 4. 完整流程总结
    show_rlhf_pipeline()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n关键要点：")
    print("1. 奖励模型学习给好回答高分（用比较数据训练）")
    print("2. KL 散度防止模型偏离太远")
    print("3. PPO 用奖励信号优化 LLM")
    print("4. RLHF 让模型更符合人类偏好")


if __name__ == "__main__":
    main()
