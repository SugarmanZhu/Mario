# 超级马里奥兄弟 RL 智能体 (PPO)

<div align="center">

**[English](README.md) | [简体中文](README.zh.md)**

![banner](assets/banner.png)

### 1-1关通关演示

![Mario 1-1 Demo](assets/demo-1-1.gif)

*使用PPO训练约1250万步后的效果*

</div>

---

一个使用 **近端策略优化 (PPO)** 学习玩超级马里奥兄弟的强化学习智能体。这是我之前基于DQN的马里奥智能体的重新实现，现在使用PPO以获得更好的稳定性和样本效率。v2.0 引入了 **IMPALA CNN** 架构和 **RGB 观测**，实现稳健的多关卡训练。

## 版本兼容性

> [!WARNING]
> **破坏性更新**
>
> | 版本 | 观测 | CNN | 动作空间 | 兼容模型 |
> |------|------|-----|---------|----------|
> | v1.x | 84×84 灰度 | NatureCNN | 7个动作 | 仅v1.x |
> | v2.x | 128×120 RGB | IMPALA | 12个动作 | 仅v2.x |
>
> **不同主版本的模型不兼容**，因为架构不同。
>
> - v2.0: 新增 `down`（进入水管）、`up`（爬藤蔓）、完整的向左移动动作、RGB观测、IMPALA CNN、用于多关卡训练的熵衰减

## 训练好的模型

| 关卡 | 版本 | 模型 | 训练步数 | 结果 |
|------|------|------|---------|------|
| 1-1 | v1.0.0 | [下载](https://github.com/SugarmanZhu/Mario/releases/download/v1.0.0/1-1-v0.zip) | ~1250万 | 通关 |
| 1-2 | v1.1.0 | [下载](https://github.com/SugarmanZhu/Mario/releases/download/v1.1.0/1-2-v0.zip) | ~970万 | 到达传送门 |

> [!NOTE]
> v1.x 模型需要 [v1.1.0](https://github.com/SugarmanZhu/Mario/releases/tag/v1.1.0) 或更早版本的代码。
> 预训练模型请查看 [Releases](https://github.com/SugarmanZhu/Mario/releases)。

## 动作空间

<details>
<summary><b>v2.x/v3.x - COMPLEX_MOVEMENT (12个动作)</b> - 当前版本</summary>

| # | 动作 | 描述 |
|---|------|------|
| 0 | `NOOP` | 不动 |
| 1 | `right` | 向右走 |
| 2 | `right + A` | 向右跳 |
| 3 | `right + B` | 向右跑 |
| 4 | `right + A + B` | 向右冲刺跳 |
| 5 | `A` | 原地跳 |
| 6 | `left` | 向左走 |
| 7 | `left + A` | 向左跳 |
| 8 | `left + B` | 向左跑 |
| 9 | `left + A + B` | 向左冲刺跳 |
| 10 | `down` | 蹲下 / 进入水管 |
| 11 | `up` | 爬藤蔓 / 进入门 |

</details>

<details>
<summary><b>v1.x - SIMPLE_MOVEMENT (7个动作)</b> - 旧版本</summary>

| # | 动作 | 描述 |
|---|------|------|
| 0 | `NOOP` | 不动 |
| 1 | `right` | 向右走 |
| 2 | `right + A` | 向右跳 |
| 3 | `right + B` | 向右跑 |
| 4 | `right + A + B` | 向右冲刺跳 |
| 5 | `A` | 原地跳 |
| 6 | `left` | 向左走 |

</details>

## 为什么选择PPO而不是DQN？

| 方面 | DQN | PPO |
|------|-----|-----|
| **稳定性** | 可能不稳定，对超参数敏感 | 通过裁剪目标实现更稳定的训练 |
| **样本效率** | 需要大型回放缓冲区 | 在线策略，无需回放缓冲区 |
| **并行化** | 单环境 | 易于跨多个环境并行化 |
| **收敛性** | 可能震荡或发散 | 更平滑、更可靠的收敛 |

## 特性

- **多关卡训练**：同时在多个关卡上训练，防止灾难性遗忘
- **IMPALA CNN架构**：约400万参数的残差CNN，更好的多任务学习
- **RGB观测**：128×120 RGB（非灰度）- 智能体可以看到HUD文字、道具颜色、关卡背景
- **熵衰减**：自动从0.08衰减到0.01，实现探索到利用的过渡
- **并行训练**：使用 `SubprocVecEnv` 实现真正的多进程并行
- **内存高效**：uint8观测配合即时归一化（32个环境约11GB滚动缓冲区）
- **自动检查点**：每10万步保存模型，支持断点续训
- **最佳模型追踪**：自动保存评估表现最好的模型
- **TensorBoard集成**：实时训练可视化
- **视频录制**：将游戏过程录制为 GIF/MP4/WebP
- **策略崩溃检测**：自动检测并恢复

## 架构 (v2.0)

### 观测处理流程

```
NES帧 (256×240 RGB)
    ↓
缩放到 128×120 (半分辨率，保持宽高比)
    ↓
帧堆叠 (4帧)
    ↓
转置为通道优先 (12, 120, 128) uint8
    ↓
即时归一化为 float32 [0, 1] (仅GPU)
```

### IMPALA CNN

```
输入: (12, 120, 128) - 4帧RGB堆叠
    ↓
阶段1: Conv3×3(12→16) → MaxPool3×3(s=2) → 2× ResBlock
    ↓
阶段2: Conv3×3(16→32) → MaxPool3×3(s=2) → 2× ResBlock
    ↓
阶段3: Conv3×3(32→32) → MaxPool3×3(s=2) → 2× ResBlock
    ↓
展平 → Linear(512) → ReLU
    ↓
策略头: Linear(256) → Linear(256) → 12个动作
价值头: Linear(256) → Linear(256) → 1个价值
```

**为什么选择IMPALA而不是NatureCNN？**
- 残差连接防止梯度消失
- 更好的多任务学习（可以通过视觉区分关卡）
- 约400万参数（与NatureCNN相近）

### 为什么选择RGB而不是灰度？

| 特征 | 灰度 | RGB |
|------|------|-----|
| HUD文字 ("WORLD 1-1") | 难以辨认 | 清晰 |
| 道具 (红/绿蘑菇) | 颜色相同 | 可区分 |
| 关卡背景 | 相似 | 每个世界独特 |
| 文件大小 | 更小 | 大3倍 |

对于多关卡训练，RGB提供关键的视觉上下文，帮助智能体识别当前关卡。

## 硬件要求

| 组件 | 最低 | 推荐 (32个环境) |
|------|------|----------------|
| GPU显存 | 4GB | 4GB (模型很小) |
| 内存 | 32GB | 64GB |
| CPU | 8核 | 16+核 |

**32个并行环境的实际使用情况：**
- GPU: ~4GB显存，~78%利用率
- 内存: 稳定~37GB，更新时峰值64GB
- CPU: ~50%利用率

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

# 训练智能体（--env 支持简写：'1-1' = 'SuperMarioBros-1-1-v0'）
python train_ppo.py --timesteps 20000000 --n-envs 32

# 多关卡训练（推荐用于泛化）
python train_ppo.py --env "1-1,1-2" --timesteps 20000000 --n-envs 32

# 观看训练好的智能体游玩
python train_ppo.py --mode play --model ./mario_models/1-1/best/best_model.zip --slow

# 录制游戏视频
python record_video.py --model ./mario_models/1-1/best/best_model.zip --output demo.gif
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
├── callbacks.py        # 训练回调（崩溃检测、熵衰减）
├── networks.py         # IMPALA CNN架构
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
# 单关卡
python train_ppo.py --env 1-1 --timesteps 10000000 --n-envs 16

# 多关卡（推荐）
python train_ppo.py --env "1-1,1-2" --timesteps 20000000 --n-envs 32
```

从检查点恢复：
```bash
python train_ppo.py --resume ./mario_models/multi-1-1-1-2/checkpoints/.../checkpoint.zip --env "1-1,1-2" --timesteps 20000000
```

### 多关卡训练

同时在多个关卡上训练，防止灾难性遗忘：
```bash
# 同时训练1-1和1-2（支持简写）
python train_ppo.py --env "1-1,1-2" --timesteps 15000000

# 恢复多关卡训练，使用更高的熵系数增加探索
python train_ppo.py --resume ./mario_models/1-2/checkpoints/.../checkpoint.zip --env "1-1,1-2" --timesteps 15000000 --ent-coef 0.07
```

多关卡训练特点：
- 将并行工作进程分配到各关卡（轮询方式）
- 在所有关卡上评估以检测遗忘
- 保存到 `mario_models/multi-1-1-1-2/` 文件夹

### 游玩模式

观看训练好的智能体游玩：
```bash
# 单关卡
python train_ppo.py --mode play --model ./mario_models/1-1/best/best_model.zip --slow

# 多关卡（依次游玩每个关卡一次）
python train_ppo.py --mode play --model ./mario_models/multi-1-1-1-2/best/best_model.zip --env "1-1,1-2" --slow
```

## 命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--mode` | `train` | `train` 或 `play` |
| `--env` | `1-1` | 环境ID，支持简写（如 `1-1`、`1-1,1-2`） |
| `--timesteps` | `2000000` | 总训练步数 |
| `--n-envs` | `16` | 并行环境数（64GB内存推荐32） |
| `--model` | `None` | 游玩模式的模型路径 |
| `--resume` | `None` | 恢复训练的检查点路径 |
| `--lr` | `0.0001` | 学习率 |
| `--ent-coef` | `0.08` | 初始熵系数（自动衰减到0.01） |
| `--n-steps` | `4096` | 每个环境每次更新的步数 |
| `--batch-size` | `256` | PPO更新的小批量大小 |
| `--slow` | `False` | 减慢播放速度便于观看 |

## PPO超参数

```python
# 环境
observation_shape = (12, 120, 128)  # 4帧RGB堆叠（通道优先）
action_space = 12                    # COMPLEX_MOVEMENT

# PPO
n_steps = 4096          # 每个环境每次更新的步数
batch_size = 256        # 小批量大小
n_epochs = 10           # 每次更新的轮数
gamma = 0.99            # 折扣因子
gae_lambda = 0.95       # GAE lambda
clip_range = 0.2        # PPO裁剪参数
ent_coef = 0.08 → 0.01  # 熵系数（线性衰减）
vf_coef = 0.5           # 价值函数系数
max_grad_norm = 0.5     # 梯度裁剪
learning_rate = 1e-4    # 使用线性衰减

# 网络
cnn = "IMPALA"          # 残差CNN
stage_depths = (16, 32, 32)
cnn_output_dim = 512
policy_net = [256, 256]
value_net = [256, 256]
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

多关卡训练 (1-1 + 1-2) 的预期行为：

| 步数 | 预期行为 | 熵 |
|------|---------|-----|
| 0-100万 | 随机探索，学习向右移动 | 0.08 → 0.07 |
| 100万-300万 | 开始躲避基本障碍 | 0.07 → 0.05 |
| 300万-700万 | 学会跳过水管和缺口 | 0.05 → 0.03 |
| 700万-1200万 | 偶尔能通关 | 0.03 → 0.02 |
| 1200万-2000万 | 稳定通关，精细调整 | 0.02 → 0.01 |

**训练速度：** 在RTX 5090上使用32个环境约672步/秒（2000万步约8小时）

## 故障排除

### 策略崩溃
如果马里奥停止移动或总是执行相同动作：
1. 运行 `python diagnose_policy.py --model 你的模型`
2. 查找健康检查点：`python diagnose_policy.py --find-healthy`
3. 从健康检查点恢复：`python train_ppo.py --resume 健康检查点`

`PolicyCollapseCallback` 在训练过程中会自动检测并从崩溃中恢复。

### 内存不足 (RAM)
如果训练崩溃或系统无响应：
- 减少 `--n-envs`（16个约20GB，32个约40GB）
- 减少 `--n-steps`（2048代替4096可将缓冲区大小减半）

### 训练卡住
`PolicyCollapseCallback` 已修复以避免与 `SubprocVecEnv` 的死锁。如果遇到卡住问题，请确保使用最新版本。

### 多关卡遗忘
如果智能体在一个关卡表现良好但忘记了另一个：
- 将 `--ent-coef` 增加到0.1以增加探索
- 确保环境在各关卡间均匀分布（轮询自动实现）
- 检查TensorBoard中的每关卡指标

## 许可证

MIT

## 致谢

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - 马里奥环境
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO实现
- [nes-py](https://github.com/Kautenja/nes-py) - NES模拟器
