动机和背景：现有的逆向规划（Inverse Planning）和目标推断模型通常假设协助者的目标是最小化主智能体（Principal Agent）达成目标的成本和通信成本问题。但是，即一定程度的次优探索对智能体的学习是有益的，协助者（Robot）应当在“允许学习”和“防止灾难/枯燥”之间进行权衡。



任务描述：网格场景。agent需要从其他地方取物体到对应地方。每个格子用有一个多维向量代表cost和risk并且有噪音（agent可以通过神经网络或贝叶斯学习者去学习这个cost和risk是多少。这个向量可以表示成感知。具体的cost和risk必须要去到才知道。robot允许一定阈值内的risk积累）

可以对agent施加的干预：开捷径（允许控制一些门的开启状态），告知（使得agent切换行动路径。这里一个点可以是，通过告知，agent会知道自己即将行动的路径上存在有大风险的路段，这个可以让agent运用RSA学习哪个路段会比较可疑有风险），提供道具（agent使用后可以无视一个或几个网格的风险）

可能的指标内容：

1. 有限时间：根据agent能力判断，到什么时间的时候要提供帮助避免超时。有限时间完成率。

2. 评估这个过程对用户是boring时候：对于不同训练程度的agent（代表不同认知能力，比如对于cost和risk的清晰程度），平衡效率和学习。预期信息增益极低且成本持续消耗。平均时间内agent的表现提升。

3. agent的表现提升（在无robot环境下规划的cost和risk。因为agent本身可能需要一些预训练）。

建模方式：

1. agent是partial observation（POMDP）。Robot需要维持一个关于“Agent知道什么（Belief over Agent's Belief）”的分布（知道agent的partial observation程度）
2. Agent是“有限理性（Boundedly Rational）”的规划方式。Robot的规划器应该包含对Agent行为的预测模型，例如使用序列逆向规划来预测Agent下一步的动作分布。





一些不确定的设定：

1. robot是否完全需要知道agent目的？是单一目的比较好还是组合目的（比如场景有多个物体，但是agent可能需要特定的两个物体的组合。robot知道有哪些组合是合理的，但是不知道agent这次是想要什么。）

2. 是否要在网格中加入agent知道但是robot不知道的向量作为“诱惑”（允许robot学习）。作为plan和goal干扰。













## Research Proposal: Balancing Goal Assistance and Agent Learning via Pedagogical Inverse Planning

### 1. Motivation and Background

Existing models of Inverse Planning and Goal Inference in cooperative environments typically operate under the assumption that the assistant's (e.g., a Robot's) primary objective is to strictly minimize the principal agent's physical and communicative costs. However, this assumption overlooks a crucial aspect of cognitive development and skill acquisition: a certain degree of suboptimal exploration is highly beneficial, and often necessary, for an agent's learning process.

Instead of acting as a pure cost-minimizer, the assisting Robot should act pedagogically. It must continuously calculate a trade-off between "allowing learning through exploration" and "preventing critical failures or cognitive fatigue (boredom/frustration)." We want to develop a framework where the Robot modulates its assistance to optimize the agent's long-term learning while maintaining safety bounds.

### 2. Task Description

The scenario is formalized as a partially observable grid-world environment where an Agent is tasked with fetching objects and delivering them to specified target locations.

- **Environment Dynamics:** Each grid cell contains a multi-dimensional latent vector representing implicit Cost and ****Risk****.
- **Perception and Learning:** The Agent receives noisy perceptual cues representing these vectors. The true cost and risk are only revealed upon physical visitation. The Agent continuously updates its internal model of these vectors using neural networks or a Bayesian learning mechanism.
- **Safety Constraints:** The Robot monitors the Agent's trajectory and permits risk accumulation only up to a predefined safety threshold, intervening when necessary.

### 3. Modalities of Robot Intervention

The Robot can intervene in the Agent's decision-making process through three primary mechanisms:

- **Environmental Modification (Shortcuts):** Altering the physical topology of the grid, such as unlocking specific doors to create safer or shorter paths.
- **Pragmatic Communication (Informing):** Providing linguistic or signal-based warnings to divert the Agent from high-risk paths. Modeled via the Rational Speech Act (RSA) framework, the Agent can pragmatically infer which specific upcoming segments are dangerous based on the Robot's utterance, thereby updating its internal risk distribution.
- **Affordance Provision (Item Drops):** Providing the Agent with consumable tools or items that temporarily grant immunity to grid risks, allowing safe traversal through otherwise hazardous areas.

### 4. Proposed Evaluation Metrics

To quantitatively evaluate the success of the pedagogical assistance framework, we propose the following metrics:

1. **Time-Bounded Success Rate:** Evaluates the Robot's ability to accurately assess the Agent's competence and provide timely interventions to prevent task timeouts.
2. **Epistemic-Cost Trade-off (Frustration/Boredom Index):** A metric to quantify when the learning process becomes counterproductive. It measures the balance between efficiency and learning for agents at different cognitive stages. "Boredom" or "Frustration" is defined as a state where the Agent's expected information gain approaches zero while physical/time costs continue to accumulate.
3. **Agent Performance Delta (Transfer Learning):** Measures the improvement in the Agent's standalone planning efficiency (cost and risk minimization) in a zero-shot, robot-free environment after interacting with the assisting Robot.

### 5. Computational Modeling Approach

**Agent Representation:** Modeled as operating within a Partially Observable Markov Decision Process (POMDP). The Agent maintains beliefs about the environment's cost/risk vectors.

**Robot Representation (Nested Theory of Mind):** The Robot must maintain a "Belief over the Agent's Belief"  to accurately gauge what the Agent currently knows or perceives regarding the environment.

**Bounded Rationality:** The Agent is modeled as a Boundedly Rational planner. The Robot utilizes a predictive model based on sequential Bayesian inverse planning  to forecast the Agent's next-action distributions and assess whether an intervention is warranted.

### 6. Design Considerations

**Q1: Degree of Goal Observability (Single vs. Compositional Goals)**

Consideration: Should the Robot have perfect knowledge of the Agent's goal? It may be more realistic and mathematically interesting to formulate the environment with Compositional Goals. For example, the Agent might need a specific combination of two items. The Robot knows the valid combinations but maintains a belief distribution over which specific combination the Agent is currently pursuing, updating this distribution as the Agent navigates.

**Q2: Information Asymmetry and Hidden Preferences ("Temptations")**

Consideration: Introducing localized reward vectors (or "temptations") that are known to the Agent but *hidden* from the Robot. This introduces a dual-learning dynamic: while the Robot is guiding the Agent, it must also perform inverse reinforcement learning  to infer the Agent's hidden preferences based on its seemingly sub-optimal deviations from the expected path. This adds robust noise to the goal recognition process and tests the resilience of the Robot's Theory of Mind model.







Project title

Balancing Goal Assistance and Agent Learning via Pedagogical Inverse Planning

Team members (for each member, indicate the full name)

Shitian Yang, Shenghan Zhou

General topic(s) related to the course

Proactive assistance, Understanding suboptimal behavior, Bayesian Inverse Planning & Goal Inference

Description of the project (problem statement, what kind of model do you plan to develop or implement, what data do you plan to use, & how it is related to the course)

### 1. Motivation and Background

Existing models of Inverse Planning and Goal Inference in cooperative environments typically operate under the assumption that the assistant's (e.g., a Robot's) primary objective is to strictly minimize the principal agent's physical and communicative costs. However, this assumption overlooks a crucial aspect of cognitive development and skill acquisition: a certain degree of suboptimal exploration is highly beneficial, and often necessary, for an agent's learning process.

Instead of acting as a pure cost-minimizer, the assisting Robot should act pedagogically. It must continuously calculate a trade-off between "allowing learning through exploration" and "preventing critical failures or cognitive fatigue (boredom/frustration)." We want to develop a framework where the Robot modulates its assistance to optimize the agent's long-term learning while maintaining safety bounds.

### 2. Task Description

The scenario is formalized as a partially observable grid-world environment where an Agent is tasked with fetching objects and delivering them to specified target locations.

- **Environment Dynamics:** Each grid cell contains a multi-dimensional latent vector representing implicit Cost and Risk.
- **Perception and Learning:** The Agent receives noisy perceptual cues representing these vectors. The true cost and risk are only revealed upon physical visitation. The Agent continuously updates its internal model of these vectors using neural networks or a Bayesian learning mechanism.
- **Safety Constraints:** The Robot monitors the Agent's trajectory and permits risk accumulation only up to a predefined safety threshold, intervening when necessary.

### 3. Modalities of Robot Intervention

The Robot can intervene in the Agent's decision-making process through three primary mechanisms:

- **Environmental Modification (Shortcuts):** Altering the physical topology of the grid, such as unlocking specific doors to create safer or shorter paths.
- **Pragmatic Communication (Informing):** Providing linguistic or signal-based warnings to divert the Agent from high-risk paths. Modeled via the Rational Speech Act (RSA) framework, the Agent can pragmatically infer which specific upcoming segments are dangerous based on the Robot's utterance, thereby updating its internal risk distribution.
- **Affordance Provision (Item Drops):** Providing the Agent with consumable tools or items that temporarily grant immunity to grid risks, allowing safe traversal through otherwise hazardous areas.

### 4. Proposed Evaluation Metrics

To quantitatively evaluate the success of the pedagogical assistance framework, we propose the following metrics:

1. **Time-Bounded Success Rate:** Evaluates the Robot's ability to accurately assess the Agent's competence and provide timely interventions to prevent task timeouts.
2. **Epistemic-Cost Trade-off (Frustration/Boredom Index):** A metric to quantify when the learning process becomes counterproductive. It measures the balance between efficiency and learning for agents at different cognitive stages. "Boredom" or "Frustration" is defined as a state where the Agent's expected information gain approaches zero while physical/time costs continue to accumulate.
3. **Agent Performance Delta (Transfer Learning):** Measures the improvement in the Agent's standalone planning efficiency (cost and risk minimization) in a zero-shot, robot-free environment after interacting with the assisting Robot.

### 5. Computational Modeling Approach

**Agent Representation:** Modeled as operating within a Partially Observable Markov Decision Process (POMDP). The Agent maintains beliefs about the environment's cost/risk vectors.

**Robot Representation (Nested Theory of Mind):** The Robot must maintain a "Belief over the Agent's Belief"  to accurately gauge what the Agent currently knows or perceives regarding the environment.

**Bounded Rationality:** The Agent is modeled as a Boundedly Rational planner. The Robot utilizes a predictive model based on sequential Bayesian inverse planning  to forecast the Agent's next-action distributions and assess whether an intervention is warranted.

### 6. Design Considerations

**Q1: Degree of Goal Observability (Single vs. Compositional Goals)**

Consideration: Should the Robot have perfect knowledge of the Agent's goal? It may be more realistic and mathematically interesting to formulate the environment with Compositional Goals. For example, the Agent might need a specific combination of two items. The Robot knows the valid combinations but maintains a belief distribution over which specific combination the Agent is currently pursuing, updating this distribution as the Agent navigates.

**Q2: Information Asymmetry and Hidden Preferences ("Temptations")**

Consideration: Introducing localized reward vectors (or "temptations") that are known to the Agent but *hidden* from the Robot. This introduces a dual-learning dynamic: while the Robot is guiding the Agent, it must also perform inverse reinforcement learning  to infer the Agent's hidden preferences based on its seemingly sub-optimal deviations from the expected path. This adds robust noise to the goal recognition process and tests the resilience of the Robot's Theory of Mind model.

### 7. What Data to Use
Because this project focuses on dynamic multi-agent reinforcement learning and planning, traditional static datasets are insufficient. Instead, we will construct a customized simulation environment and generate synthetic data. We will build a 2D grid-world environment containing implicit Cost and Risk dynamics from scratch. May utilize the OpenAI Gymnasium framework.

### 8. How it is Related to the Course
The principal Agent is specifically modeled as boundedly rational and prone to suboptimal exploration, requiring the Robot to interpret its mistakes not as random noise, but as a learning process. And the Robot moves beyond passive reaction to proactive, "scaffolding" assistance based on a computational Theory of Mind. The core mechanism of the Robot relies on Bayesian inverse decision-making to infer the Agent's hidden beliefs and goals.

Timeline

3.8-3.22: Literature Review & Environment Setup.

3.23-3.31: Agent Modeling & Baseline Implementation

4.1-4.7: Robot Theory of Mind & RSA Integration, Intervention Logic & Trade-off Optimization

4.8-4.20: Evaluation & Metrics Collection, Data Analysis, Ablation Studies

