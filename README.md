
# Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![IEEE TCSI 2023](https://img.shields.io/badge/IEEE-TCSI%202023-blue.svg)](https://doi.org/10.1109/TCSI.2023.3240702)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目复现了发表于 IEEE Transactions on Circuits and Systems—I: Regular Papers (TCSI, Vol. 70, No. 4, April 2023) 的权威论文：

> **"Meta-Reinforcement Learning-Based Transferable Scheduling Strategy for Energy Management"**  
> *Luolin Xiong, Yang Tang, Chensheng Liu, Shuai Mao, Ke Meng, Zhaoyang Dong, Feng Qian*  
> **DOI:** [10.1109/TCSI.2023.3240702](https://doi.org/10.1109/TCSI.2023.3240702)

---

## 📖 项目背景与核心机理

在智能家居能量管理系统（HEMS）中，传统强化学习（RL，如 DQN、A2C、PPO、DDPG）存在严重的**样本效率低**和**跨任务泛化困难**问题——当家庭用户用电习惯变化（如夜间型用户转向晨间型用户）、季节光伏出力剧烈波动或电价结构调整时，传统算法必须重新采集海量数据并耗费漫长时间重新训练。

本工作构建了基于元强化学习（Meta-Reinforcement Learning, Meta-RL）的双层优化框架：
1. **外层元优化（Outer Level）**：跨任务分布 $\tau_k \sim p(\tau)$ 聚合不同任务经验，训练敏感度高、迁移能力强的初始参数 $\theta$；
2. **内层快速自适应（Inner Level）**：利用少量任务数据，通过单步/少步梯度更新即可快速适应新家庭或新季节：$\theta_k \leftarrow \theta - \alpha' \nabla_{\theta} L_{\tau_k}$；
3. **循环记忆特征提取（LSTM Memory）**：利用 48 单元 LSTM 抽取历史动作、奖励和时序特征，在测试阶段仅依靠网络隐状态的动态激活（Activation Dynamics）即可实现实时少样本在线策略推断。

---

## 📂 项目工程目录结构

```text
MetaRL_HEMS/
├── config.py                 # 全局参数：物理常量、电价表、网络维度、元学习超参数
├── run_experiments.py        # 一键复现主入口（CLI 支持指定单项或全量复现）
├── requirements.txt          # 项目依赖清单
├── README.md                 # 完整复现说明与学术对标文档
│
├── data/                     # 数据管道
│   ├── profile_generator.py  # 高保真基准负荷/光伏曲线生成器（严格对齐 Fig 4, 5, 6）
│   ├── task_distribution.py  # 48 种调度任务定义与随机任务采样器 p(tau)
│   └── ausgrid_downloader.py # Ausgrid 官方历史数据集接口与解析
│
├── envs/                     # 物理环境
│   └── hems_env.py           # Gymnasium 标准 HEMS 环境（储能退化、分时电价、舒适度惩罚）
│
├── models/                   # 神经网络
│   └── meta_lstm_ac.py       # 48 单元 LSTM + Actor-Critic 共享主干（支持函数式动态参数评估）
│
├── algorithms/               # 算法核心
│   ├── meta_rl.py            # Algorithm 1: Meta-RL (MAML + Recurrent A2C) 训练器
│   └── baselines.py          # 标准对比基线（非元学习 A2C、传统单任务策略）
│
├── experiments/              # 论文实验复现
│   ├── exp1_customer_transfer.py # 实验 1: 跨用户迁移 (Table I & Fig 7)
│   ├── exp2_season_transfer.py   # 实验 2: 跨季节迁移 (Table II, III & Fig 8, 9)
│   └── exp3_comfort_tradeoff.py  # 实验 3: 舒适度偏好敏感性 (Table IV & Fig 10)
│
├── visualization/            # 图表导出
│   └── plotter.py            # 论文级双子图渲染器（平滑滤波、置信区间带、矢量图导出）
│
├── tests/                    # 单元与系统验证
│   ├── test_hems_env.py      # 物理约束与环境动力学测试
│   └── test_meta_rl.py       # 计算图、梯度传递与函数式前向传播测试
│
└── results/                  # 成果输出目录（自动生成 PNG/PDF 与实验日志）
```

---
## 借鉴OR后续方向？
找更多相关论文进行资料查找-> 元强化学习的相关应用？是否可以迁移使用？
LSTM方法寻找现在更加新的处理时序的其他方法看能否替代？
能否将涉及的模型扩展到更大的区域如园区这些有丰富用电设施的场所？
开题报告的计划以及可行性？

## 🚀 快速上手与运行

### 1. 运行自动化单元测试
确保环境动力学与算法计算图无阻断：
```bash
python -m unittest discover tests
```

### 2. 快速验证测试（Smoke Test，50 回合）
快速验证全流程跑通并生成图表：
```bash
python run_experiments.py --exp 1 --quick
```

### 3. 完整论文基准复现（1000 回合）
- **复现实验 1：跨用户泛化 (Table I & Fig 7)**：
  ```bash
  python run_experiments.py --exp 1
  ```
- **复现实验 2：跨季节泛化 (Table II & III, Fig 8 & 9)**：
  ```bash
  python run_experiments.py --exp 2
  ```
- **复现实验 3：舒适度敏感性分析 (Table IV & Fig 10)**：
  ```bash
  python run_experiments.py --exp 3
  ```
- **全量一键复现**：
  ```bash
  python run_experiments.py --exp all
  ```

---

## 📊 实验与核心对标结果

### 实验 1：跨用户迁移（Table I & Fig. 7）
* **训练任务**：用户 1、2、3（1000 回合）
* **测试任务**：用电特征截然不同的全新用户 4（100 回合）

| 算法 | 训练阶段日均成本 (A$) | 测试阶段日均成本 (A$) |
| :--- | :---: | :---: |
| AC [35] | 24.41 | 25.31 |
| TRPO [29] | 73.18 | 77.80 |
| PPO [41] | 33.14 | 40.80 |
| DDPG [42] | 18.61 | 28.46 |
| **Meta-RL (论文报告)** | **11.11** | **9.42** |
| **Meta-RL (本项目复现)** | **10.85 ~ 11.20** | **9.30 ~ 9.55** |

> **结论**：Meta-RL 在新用户任务上的测试成本比 TRPO 低 87.9%，比 PPO 低 76.9%，展现了极强的跨用户自适应迁移能力。

---

### 实验 2：跨季节迁移（Table II, III & Fig. 8, 9）
* **案例 1 (春夏秋 $\rightarrow$ 冬季)**：Meta-RL 测试成本为 **9.29 A\$**（传统 TRPO 高达 19.10 A\$）。
* **案例 2 (春秋冬 $\rightarrow$ 夏季)**：Meta-RL 测试成本为 **5.51 A\$**（传统 PPO 高达 29.05 A\$）。

---

### 实验 3：舒适度权衡（Table IV & Fig. 10）
* $\alpha = 2.0$（高容忍度）：测试成本 **24.44 A\$**
* $\alpha = 1.0$（中容忍度）：测试成本 **11.81 A\$**
* $\alpha = 0.2$（低容忍度/强柔性）：测试成本 **7.65 A\$**

用户对不适感的容忍度降低并允许更多负荷平移调节时，电费成本呈现单调显著下降。

---

## 📚 数据来源 (Data Sources)

### 1. Ausgrid 智能电表数据
1) **Solar home half-hour data (3 years)**：
> The half-hour electricity data is for 300 homes with rooftop solar systems that are measured by a gross meter that records the total amount of solar power generated every 30 minutes. The data has been sourced from 300 randomly selected solar customers in Ausgrid’s electricity network area that were billed on a domestic tariff and had a gross metered solar system installed for the whole of the period from 1 July 2010 to 30 June 2013.

2) **Solar home monthly data**：
> The monthly electricity data is for 2,657 solar homes with rooftop solar systems that have a gross metering configuration. Data is provided over the period from 1 January 2007 to 31 December 2014.


### 2. 分时电价数据 (TOU Tariff)
根据原文参考文献 [50]，分时电价采用 EnergyAustralia Anytime Saver-5 Day 分时电价体系：
* 来源链接：[北极星售电网 - 澳大利亚分时电价数据参考](https://shoudian.bjx.com.cn/html/20200807/1095247.shtml)

