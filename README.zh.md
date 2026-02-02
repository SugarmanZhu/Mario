# 超级马里奥兄弟 RL 智能体 (PPO)

<div align="center">

**[English](README.md) | [简体中文](README.zh.md)**

![banner](assets/banner.png)

### 1-1关通关演示

![Mario 1-1 Demo](assets/demo-1-1.gif)

*使用PPO训练约1250万步后的效果*

</div>

---

一个使用 **近端策略优化 (PPO)** 学习玩超级马里奥兄弟的强化学习智能体。这是我之前基于DQN的马里奥智能体的重新实现，现在使用PPO以获得更好的稳定性和样本效率。

## 训练好的模型

| 关卡 | 模型 | 训练步数 | 结果 |
|------|------|---------|------|
| 1-1 | [下载](https://github.com/SugarmanZhu/Mario/releases/download/v1.0.0/1-1-v0.zip) | ~1250万 | 通关 |

> 模型使用代码版本 `v1.0.0` 训练。预训练模型请查看 [Releases](https://github.com/SugarmanZhu/Mario/releases)。

## 为什么选择PPO而不是DQN？

| 方面 | DQN | PPO |
|------|-----|-----|
| **稳定性** | 可能不稳定，对超参数敏感 | 通过裁剪目标实现更稳定的训练 |
| **样本效率** | 需要大型回放缓冲区 | 在线策略，无需回放缓冲区 |
| **并行化** | 单环境 | 易于跨多个环境并行化 |
| **收敛性** | 可能震荡或发散 | 更平滑、更可靠的收敛 |

## 特性

- **并行训练**：使用 `SubprocVecEnv` 实现真正的多进程并行
- **自动检查点**：每10万步保存模型，支持断点续训
- **最佳模型追踪**：自动保存评估表现最好的模型
- **TensorBoard集成**：实时训练可视化
- **视频录制**：将游戏过程录制为 GIF/MP4/WebP
- **策略崩溃检测**：自动检测并恢复

## 使用的硬件

- **GPU**: NVIDIA GeForce RTX 5090 (32GB显存)
- **内存**: 64GB
- **系统**: Windows 11

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/SugarmanZhu/Mario.git
cd Mario

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装支持CUDA的PyTorch (RTX 40/50系列)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装依赖
pip install -r requirements.txt

# 测试环境
python test_env.py

# 训练智能体
python train_ppo.py --mode train --timesteps 10000000 --n-envs 16

# 观看训练好的智能体游玩
python train_ppo.py --mode play --model ./mario_models/flag/1-1-v0.zip --slow

# 录制游戏视频
python record_video.py --model ./mario_models/flag/1-1-v0.zip --output demo.gif
```

## 录制游戏视频

```bash
# 录制为GIF（默认）
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.gif

# 录制为MP4
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.mp4 --fps 30

# 录制为WebP（更小的文件）
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.webp --quality 90

# 录制多个回合，保留最佳
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.gif --episodes 5 --best
```

## 项目结构

```
Mario/
├── train_ppo.py        # 主训练和推理脚本
├── record_video.py     # 录制游戏为 GIF/MP4/WebP
├── wrappers.py         # 自定义环境包装器
├── callbacks.py        # 训练回调（崩溃检测）
├── utils.py            # 工具函数
├── diagnose_policy.py  # 策略健康分析工具
├── requirements.txt    # Python依赖
├── mario_models/       # 保存的模型检查点
│   ├── best/           # 基于评估的最佳模型
│   └── flag/           # 能通关的模型
└── mario_logs/         # TensorBoard日志
```

## 训练

开始新的训练：
```bash
python train_ppo.py --mode train --timesteps 10000000 --n-envs 16
```

从检查点恢复：
```bash
python train_ppo.py --mode train --resume ./mario_models/checkpoint.zip --timesteps 20000000
```

## 命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--mode` | `train` | `train` 或 `play` |
| `--env` | `SuperMarioBros-1-1-v0` | 环境ID |
| `--timesteps` | `2000000` | 总训练步数 |
| `--n-envs` | `16` | 并行环境数 |
| `--model` | `None` | 游玩模式的模型路径 |
| `--resume` | `None` | 恢复训练的检查点路径 |
| `--lr` | `0.0001` | 学习率 |
| `--slow` | `False` | 减慢播放速度便于观看 |

## PPO超参数

```python
n_steps = 2048          # 每个环境每次更新的步数
batch_size = 512        # 小批量大小
n_epochs = 10           # 每次更新的轮数
gamma = 0.99            # 折扣因子
gae_lambda = 0.95       # GAE lambda
clip_range = 0.2        # PPO裁剪参数
ent_coef = 0.05         # 熵系数
vf_coef = 0.5           # 价值函数系数
max_grad_norm = 0.5     # 梯度裁剪
learning_rate = 1e-4    # 使用线性衰减
```

## 自定义奖励塑形

| 奖励 | 数值 | 描述 |
|------|------|------|
| 前进奖励 | `+0.1 × Δx` | 向右移动的奖励 |
| 停滞惩罚 | `-0.5` | 10帧以上不移动 |
| 时间惩罚 | `-0.01` | 每步（鼓励速度） |
| 通关奖励 | `+100` | 到达旗杆 |
| 死亡惩罚 | `-50` | 未通关死亡 |
| 分数奖励 | `+0.01 × Δ分数` | 消灭敌人、获取道具 |
| 金币奖励 | `+5.0` 每个 | 收集金币 |

## 训练进度

| 步数 | 预期行为 |
|------|---------|
| 0-50万 | 随机探索，学习向右移动 |
| 50万-200万 | 开始躲避基本障碍 |
| 200万-500万 | 学会跳过水管和缺口 |
| 500万-1000万 | 偶尔能通关1-1 |
| 1000万+ | 稳定通关 |

## 故障排除

### 策略崩溃
如果马里奥停止移动或总是执行相同动作：
1. 运行 `python diagnose_policy.py --model 你的模型`
2. 查找健康检查点：`python diagnose_policy.py --find-healthy`
3. 从健康检查点恢复：`python train_ppo.py --resume 健康检查点`

### 训练卡住
`PolicyCollapseCallback` 已修复以避免与 `SubprocVecEnv` 的死锁。如果遇到卡住问题，请确保使用最新版本。

## 许可证

MIT

## 致谢

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - 马里奥环境
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO实现
- [nes-py](https://github.com/Kautenja/nes-py) - NES模拟器
