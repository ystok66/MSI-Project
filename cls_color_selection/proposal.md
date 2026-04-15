基于CLS的learner模型。
新场景流程：
1. LEARNER预先学习n_sup样本。
可选：(tutor先观测几轮learner冻结长期学习参数的测试情况。为n_obs次，默认4次)
2. tutor 观察并且干预learner n_q样本
3. learner冻结长期学习参数，不会更新，测评n_test样本

新场景描述：
给task文字描述，要求组合出对应的小球顺序。
比如lug fep blicket, 要回答出YELLOW PURPLE这样。
给候选框，候选框里面有n_choice个小球，小球颜色种类是来自于grammar里面有哪几种。并且是随机抽取，可以有多个相同的，也可以有某种颜色这次没有。
learner需要把若干个小球放learner应该放的位置上，参考FAQ 2.
learner可以选择点击confirm，表示提交答案。也可以选择retry表示再次抽取候选框的小球。
设置一个n_confirm_max，表示最多可以点击confirm的次数。如果超过了，就失败。
learner应该根据自己效用或者是参数，从候选框取若干个小球
每个球同时有一个vector，代表了risk。这个risk维度是n_vector(默认10).其中有n_safe种类（默认5）是代表safe的vector，n_danger种类（默认5）是代表danger的vector。在候选框里的小球有risk的是para_r/n_choice(para_r默认是0.5). 且给learner看到的vector是会加上一些小噪音的.但是tutor是知道这个vector的是否有risk的。

若干learner选取到有risk的小球，放到list里面，learner就会死亡。默认失败。这里的次序是，learner选取了小球，然后等tutor反馈，如果tutor是wait，则放到list里面，如果是其他情况，参照tutor的说明。但是learner死亡时候，是可以知道这个小球的vector是danger的。

tutor维持一个CLS EM模型，代表对learner的建模。每次learner尝试抽取摆放小球，tutor会更新信念（为了避免每次都更新，是当learner再次retry或者是confirm时候，才会更新。避免每次放入小球都更新太频繁）。

tutor可以选择warning。就是发现learner选取了有risk的小球，但是还没有放到list里面，tutor可以给一个warning，让learner知道这里面有risk的小球，但是learner并不知道具体是哪个小球有risk。这个可以提供集合条件化的贝叶斯更新来进行学习到risk。（注意：这里不是需要根据tutor的行为来集合条件化的贝叶斯更新，而是根据warning来集合条件化的贝叶斯更新。因为warning了，但是learner不知道select的若干个小球里面，有几个是danger的，哪几个是。所以需要通过集合条件化的贝叶斯更新来学习哪些是danger的，学习到几类不同的danger vector类型。）

tutor也可以选择hint。hint的意思是，tutor可以在list任意位置放上若干小球，使得learner能加速，避免超时。tutor是可以根据：
tutor 估计 P_success_before_timeout
分别在 WAIT / HINT_1 / HINT_2 / ... 下做 counterfactual estimate
选 argmax_a Q(a)

其中 Q(a) 里把：

eval gain
teach success gain
death reduction
timeout reduction
over-help penalty

全放进去。
需要考虑到risk小球概率，有多少个小球，每个颜色小球出现的概率，剩余需要多少个小球。

learner应该根据hint max P(hint_positions|knowledge)这样。

Tutor的效用可以由：
1. 最大化learner的学习变化（eval）。这里可以是信念里面，learner的参数分布和基于这个grammar的最正确的参数分布的KL散度。也可以是提供类似mcmc采样随机一些题目来看看learner的eval变化。提高learner在后期eval的泛化效果。
2. learner的死亡率。死亡率越低越好。
3. learner的在tutor干预过程中的失败率，失败率越低越好。
4. learner的平均confirm次数。越少越好。

这里特别的，可能由不同种类的tutor，比如有些tutor可能注重learner在后期eval的泛化效果而不注重learner的在tutor干预过程中的失败率和learner的平均confirm次数。有些是可能注重learner的在tutor干预过程中的失败率，但是只要保证learner在后期eval不至于太差。但是无论如何，都要尽可能避免死亡。

learner的效用可以是：
1. 后期eval的泛化效果
2. 避免死亡
3. 平均confirm次数
4. learner的在tutor干预过程中的失败率
这里也是，可以有不同种类的learner的感觉。

为了保证测试，或者说整个场景是正常工作的。评测是两个方面：
有一种tutor是目的是最大化learner在后期eval的泛化效果。这个tutor允许learner的在tutor干预过程中的失败率的存在。
另一种是保证learner的在tutor干预过程中的失败率比较低，但是尽可能维持住最大化learner在后期eval的泛化效果。
需要记录的有：teach干预过程的平均死亡次数，超时次数比例，平均confirm次数(成功案例的)。还有eval过程中的死亡率、成功率、超时次数比例。

作为对比，有三种参照组：
1. 一个是no tutor的，跑一样的n q，如果死亡就直接判定失败。因为learner死亡时候也能学到risk。看看学习过程中（因为是no tutor，这个过程对应的是teach干预过程）的平均confirm次数(成功案例的)，超时次数比例：eval时候的死亡率、成功率、超时次数比例。
2. 也是no tutor的（no_tutor_immortal_warnlike），跑一样的n q，但是不死亡。就是它可以知道自己要死亡（相当于tutor warning，可以通过集合条件化的贝叶斯更新知道哪个danger），然后继续尝试，依旧有n_confirm_max。看看在学习过程中（因为是no tutor，这个过程对应的是teach干预过程）的平均TeachDangerSelectCount，平均confirm次数(成功案例的)，eval时候的死亡率、成功率、超时次数比例。
3. 也是no tutor的（no_tutor_immortal_no_timeout_upperbound），跑一样的n q，但是不死亡，同时不会有超时。看看在学习过程中（因为是no tutor，这个过程对应的是teach干预过程）的平均TeachDangerSelectCount，平均confirm次数(成功案例的)，eval时候的死亡率、成功率、超时次数比例（这里的eval是有超时的）。


我觉得可以分成三个阶段走
1. 基础场景和learner模型
2. tutor模型
3. tutor+learner一起

FAQ：
1. 最核心的不明确：CLS 在这个新场景里到底“学什么”
在预先学习阶段，直接看若干个文本和小球序列对。在tutor学习阶段，学习可以是：warning带来的risk学习；confirm错误带来的负反馈；hint之后的最大化。

可以设置config
基础是：只知道错。
另一种是：知道哪些位置错。
但是要求learner能根据错误，或者说知道自己这个是错的，能更新自己的模型。

{
    confirm 错误后的 grammar 更新：我给你两种都能写进当前 CLS 的贝叶斯版本

你当前 CLS 数学里已经有现成骨架了：对每个候选 trace (\pi_k)，先算分数

[
\mathcal L(\pi_k)=\ln P(\pi_k\mid U,\Phi)+\ln P(Y\mid \mathrm{Exec}(\pi_k))+\ln P_{S_1}(U\mid \mathrm{Exec}(\pi_k))
]

然后做 soft responsibilities，再做 M-step 更新 role counts 和 emission 的充分统计量。你的文档里本来就把 inner-loop 写成了 beam search 的 E-step 加 soft-EM 的 M-step，这正适合继续走“责任重加权”的路子，而不是硬改成梯度法。CLS cortex 里的 role counts 也是 Dirichlet-smoothed counts，发射部分是 NIG posterior，这类参数天然适合走共轭/计数式更新。Dirichlet 对 categorical / multinomial 的共轭更新、Beta 对 Bernoulli 的共轭更新，本质上都是“旧参数 + 伪计数/新证据”的形式。  ([Brown University Computer Science][1])

我建议你把 confirm feedback 统一写成：

### 总框架

先用当前模型对 top-(K) trace 做基础 posterior：

[
q_k ;=; P(\pi_k\mid U,D_t)
;\propto;
\exp\big(\mathcal L(\pi_k)\big)
]

其中 (D_t) 是 learner 目前的长期知识。

confirm 之后，不直接“喂真答案”，而是引入一个**反馈似然**：

[
P(F \mid \hat Y, Y_k)
]

这里

* (\hat Y) 是 learner confirm 的答案
* (Y_k=\mathrm{Exec}(\pi_k)) 是第 (k) 个候选 trace 的输出
* (F) 是 confirm 后拿到的反馈

然后把 posterior 改成：

[
\tilde q_k
;\propto;
q_k \cdot P(F \mid \hat Y, Y_k)
]

最后仍然沿用你原来 CLS 的 M-step，只是把责任从 (q_k) 换成 (\tilde q_k)。
为了避免同一个 query 被“重复当新监督样本硬塞进去”，我建议用**差分更新**，而不是再完整加一遍：

[
n_{w,r}^{,new}
==============

n_{w,r}^{,old}
+
\eta_{fb}
\sum_{k=1}^K
(\tilde q_k-q_k),
C_{w,r}(\pi_k)
]

这里 (C_{w,r}(\pi_k)) 是 trace (\pi_k) 中词 (w) 取 role (r) 的次数。
对于 emission / repeat 的充分统计量，也同样做：

[
S_w^{,new}
==========

S_w^{,old}
+
\eta_{fb}
\sum_{k=1}^K
(\tilde q_k-q_k),
T_w(\pi_k)
]

然后再用你现有的 NIG / moment-matching 更新去还原 ((\mu,\Sigma,\kappa)) 等参数。你自己的文档里本来就已经把连续概念更新写成了 online moment matching，这里直接沿用就行。 

---

### 2.1 方案 A：只知道“错了”

这是最基础的 negative evidence。

此时反馈事件是：

[
F_{\text{wrong}} = {\text{true output} \neq \hat Y}
]

对每个候选 trace (\pi_k)，定义它与 learner 提交答案的接近度。
最自然的是复用你现在已有的 soft alignment / soft edit distance 思路：

[
d_k = d(\hat Y, Y_k)
]

然后定义

[
P(F_{\text{wrong}} \mid \hat Y, Y_k)
====================================

1-\exp(-\beta_{\text{err}}, d_k)
]

解释很直观：

* 如果 (Y_k) 和 learner 提交的 (\hat Y) 很像，尤其完全一样，那么它应该被 confirm 的“错了”明显打压
* 如果 (Y_k) 和 (\hat Y) 差很远，那它更可能还活着

在离散 one-hot 颜色、且只做 exact match 的简化版里，可以直接写成：

[
P(F_{\text{wrong}} \mid \hat Y, Y_k)=
\begin{cases}
\varepsilon_{\text{wrong}}, & Y_k=\hat Y[4pt]
1-\varepsilon_{\text{wrong}}, & Y_k\neq \hat Y
\end{cases}
]

其中 (\varepsilon_{\text{wrong}}) 很小，比如 (10^{-3}) 到 (10^{-2})。

然后：

[
\tilde q_k
==========

\frac{q_k,P(F_{\text{wrong}}\mid \hat Y,Y_k)}
{\sum_j q_j,P(F_{\text{wrong}}\mid \hat Y,Y_j)}
]

最后把 (\tilde q_k) 带回上面的差分 M-step。

这个方案的含义是：

* 你没有学到“正确答案是什么”
* 但你学到了“当前这类会导出 (\hat Y) 的解释，不对”

所以它是纯负证据更新。

---

### 2.2 方案 B：知道哪些位置错

这时反馈更强，因为它同时给了**部分正证据**和**部分负证据**。

设反馈给出一个 mask：

[
m_\ell \in {0,1},\qquad \ell=1,\dots,L
]

其中

* (m_\ell=1)：第 (\ell) 位是对的
* (m_\ell=0)：第 (\ell) 位是错的

对候选 trace (\pi_k) 的输出 (Y_k=(y_{k,1},\dots,y_{k,L}))，定义每一位的匹配概率：

离散版可写为

[
s_{k,\ell}
==========

# P(\text{match at }\ell \mid y_{k,\ell},\hat y_\ell)

\begin{cases}
1-\varepsilon_{eq}, & y_{k,\ell}=\hat y_\ell[4pt]
\varepsilon_{eq}, & y_{k,\ell}\neq \hat y_\ell
\end{cases}
]

连续 Lab 版则可以写成 Gaussian/RBF 形式：

[
s_{k,\ell}
==========

\exp!\left(
-\frac{|y_{k,\ell}-\hat y_\ell|^2}{2\sigma_{eq}^2}
\right)
]

然后整体反馈似然定义成：

[
P(F_{\text{mask}} \mid \hat Y, Y_k, m)
======================================

\prod_{\ell:m_\ell=1} s_{k,\ell}
;\cdot;
\prod_{\ell:m_\ell=0} (1-s_{k,\ell})
]

这就是：

* 对“反馈说对了”的位置，鼓励候选 trace 也在那里匹配
* 对“反馈说错了”的位置，鼓励候选 trace 在那里不要匹配

于是

[
\tilde q_k
==========

\frac{q_k,P(F_{\text{mask}}\mid \hat Y,Y_k,m)}
{\sum_j q_j,P(F_{\text{mask}}\mid \hat Y,Y_j,m)}
]

再用同一个差分 M-step：

[
n_{w,r}^{,new}
==============

n_{w,r}^{,old}
+
\eta_{fb}
\sum_{k=1}^K
(\tilde q_k-q_k),
C_{w,r}(\pi_k)
]

[
S_w^{,new}
==========

S_w^{,old}
+
\eta_{fb}
\sum_{k=1}^K
(\tilde q_k-q_k),
T_w(\pi_k)
]

这个方案的好处是：
它不是简单地“把整条 wrong trace 全否掉”，而是允许你从 confirm 里学到：

* 哪些位置的 grammar 已经对了
* 哪些位置的 grammar 还错

}

2. action space：
就相当于从原材料里面拿去物品，如何拼装的感觉。当然我不确定这个list是必须要按照顺序拼还是可以分段来拼。
理论上在一个阶段里，除了hint和confirm，learner的CLS参数是不会改变的，所以其对于这个语句的输出是固定的，现在只是需要得到对应的材料，避免risk就行了。所以不会存在任意位置放、分段放、之后再补的情况。同时tutor的hint之后出现在confirm之后立即出现这样（这样规定）。
意思是说，比如现在根据文字要求，因为是CLS模型，已经生成出一个输出，比如是AAABBBCC。但是现在原料库只有BBCD。那就只能是[***BB*C*************]这样。

{
    Action space：不要再把它当“自由拼装”，而要当“固定目标下的补料”

设当前 CLS 对文本 (U) 的固定输出为

[
Y^*=(y_1,\dots,y_L)
]

例如 (Y^*=\texttt{AAABBBCC})。

当前状态不要真的存一个“任意 20 格可乱放”的 list，而是存一个**目标对齐的部分完成状态**：

[
z_t=(z_{t,1},\dots,z_{t,L}), \qquad z_{t,\ell}\in{\texttt{EMPTY}}\cup \mathcal C
]

其中只有当某个位置已经被补到正确颜色时，那个位置才被填上；否则就是 `EMPTY`。
同时存一个候选库 multiset (M_t)。

再定义每种颜色的缺口：

[
\delta_t(c)=#{\ell:y_\ell=c}-#{\ell:z_{t,\ell}=c}
]

learner 的一次动作不是“任意摆”，而是：

[
a_t = S_t \subseteq M_t
]

这里 (S_t) 是它这一次从原料库里拿的一组球，而且只允许拿**当前还缺**的颜色；如果拿了不缺的颜色，直接视为低效动作，效用设成很低，或者直接非法。

然后放置不是 learner 自己再决定坐标，而是一个**确定性放置算子**：

[
z_{t+1}=\mathrm{Place}(z_t,Y^*,S_t)
]

规则就是：对 (S_t) 里每个颜色 (c)，把它自动填到 (Y^*) 中该颜色最左边尚未完成的位置。

这样一来，你举的例子就成立了：

* 固定目标：`AAABBBCC`
* 当前原料库：`BBCD`
* learner 这轮若选 `BBC`
* 那么更新后状态就是

[
[\ast \ast \ast B B \ast C \ast]
]

也就是你写的那种 `[***BB*C*...]` 的含义。这里最重要的是：**这不是“任意位置拼装”**，而是“围绕固定目标输出的目标对齐填充”。这样就不会重新引入分段拼装、自由摆位、再覆盖之类的复杂状态。这个也和你自己现在的设想一致：除了 `hint` 和 `confirm` 之外，CLS 对这句的解释不变，变化的是材料获取和 risk 决策。 

我会建议你把 action space 直接写成这四个原子动作：

1. `SELECT(S_t)`：一次性选一组当前想拿的球
2. `WAIT/WARNING`：tutor 在 `select 后、place 前` 介入
3. `AUTO-PLACE`：若无 warning 且未触发死亡，按上面的 `Place` 规则自动填入
4. `CONFIRM`：检查当前 (z_t) 是否已满且与 (Y^*) 一致；失败后才能进入 hint 评估
}

3. retry 的语义不清楚
retry就相当于更新原料库。因为有可能有些原料我想用但是有risk，或者说我已经用完我觉得需要用的原材料了。每次select完后就会自动retry，因为每次select相当于learner一次性拿来需要的所有原材料。retry/自动刷新之后，当前 list 保留。在已有 list 上继续补（理论上在一个阶段里，除了hint和confirm，learner的CLS参数是不会改变的，所以其对于这个语句的输出是固定的，现在只是需要得到对应的材料，避免risk就行了）。warning 后当前 select 整组作废

4. confirm budget 和 step budget 的关系还没定
只有 confirm 有上限，timeout 是由 confirm 次数触发。一个 learner 可以无限试摆，只要不 confirm，因为这个CLS模型应该不会无限尝试吧，每次都是最大化P这样。因为tutor要不就是confirm之后才会介入，因为timeout 是由 confirm 次数触发，tutor只需要在某次confirm之后去选择要不要选择hint。
warning：在 select 后、place 前 介入。
hint：在 confirm 失败后。
timeout 仍然只由 confirm 次数触发。
为了防止learner太胆小，总是认为都是risk。retry大于n_retry_courage（默认5）之后，tutor会看情况，如果当前有小球是当前需要（list里面这个颜色的小球小于正确答案里面这个小球的颜色数量）且没有risk，会courage。courage时候，learner会通过集合条件化的贝叶斯更新学习这几个choice里面有safe的，直到learner第一次把若干小球放到list里面，才会重新更新n_retry_courage（warning因为是select阶段，所以warning不会更新n_retry_courage）。

5. danger 的发生时机需要再严格一点
risk 结算点 = pick 后、place 前、tutor 干预之后。tutor 选择 WAIT，而 select 集合里含 danger，一触即死，整个 query 结束

6. warning 的语义还不够封闭
warning 是针对，当learner已经select好了小球之后warning。这里先假设warning之后learner会放弃当前的select内容（实现代码时候里可以备注一下，这里可以加信任度机制的）。warning 的 集合条件化的贝叶斯更新 学习对象是：这里一共有n_safe+n_danger种类别，且这两种类别是互斥的。能不能使用Pragmatic Inference (Rational Speech Acts)的Multi-Intent Inference (Cardinality & Sets)来建模？
{
    vector 到 type 的先验

先定义 learner 对每个球的 type posterior：

[
P(z_i=k \mid x_i)
\propto
P(x_i \mid z_i=k),P(z_i=k)
]

这里 (k=0) 是 safe，(k=1,\dots,K) 是不同 danger type。
这个可以直接用你喜欢的贝叶斯方式，比如：

* 每类一个 Gaussian prototype
* safe / danger type 用 Dirichlet-Categorical 或 Gaussian mixture
* online moment matching / conjugate update
}

warning 是永远 truthful. tutor不会因为教学目的，故意不 warning,因为避免死亡是重中之重。tutor policy不会存在warning的“策略性沉默”。务必上上完整的 Multi-Intent Cardinality & Sets。

7. hint 的动作空间还没界定
先使用贪心算法，目的是最大化效用。比如我放入一个球，是优先考虑time out还是eval的学习。如果一个球不够，还是time out可以选择再放一个。最佳的情况是避免time out同时保证eval。这里可以看看能不能使用smc或者是mcmc来？
只有 retry/confirm checkpoint 更新 grammar belief。
hint 放进去的球是否默认 safe。
hint 放的球是“凭空添加”
learner 能不能覆盖 tutor hint：但是理论上需要learner进行最大化P（model|hint），hint 球进入 list 后就当作已放置事实，不再允许覆盖。

risk belief：在 select 后可更新
grammar belief：只在 retry/confirm checkpoint 更新

8. 你写的 hint 判据，目前公式方向上还不够自洽
这里我想表示的是，tutor能根据当前对learner的估计（预计需要多少次confirm才能完成的期望），当然这里的公式如何融入到效用里面是一个问题。hint 本身不会消耗 confirm 机会。

9. tutor belief update 的观测边界还没完全定
A. tutor 在一个 query 内能看到什么：learner 当前 list，learner 的pick
B. tutor 更新的是哪一层 belief：当learner select时候，像之前说的，tutor可以选择warning，这时候是更新risk估计。当learner摆放小球到list时候，retry或者是confirm之前那一step，tutor会进行更新（相当于最大化P（当前场景|当前观测））估计，即grammar knowledge belief。

10. learner 的“长期冻结”与 risk 学习是否也冻结，没写清楚
grammar 参数冻结
risk 头冻结
query 内短期 memory 允许存在
tutor 不在时，learner 死亡后也不更新

11. baseline 2 和 baseline 3 的定义里有点互相打架
如果它真的“不死亡”，那 teach 过程里的死亡次数应该恒为 0：应该是说记录其选了risk小球的次数。
但后面又想记录“超时次数比例”。虽然学习阶段没有，但是eval阶段会有。

12. 你说“不同种类 tutor / learner 有不同偏好”，但还没决定是固定 config 还是 latent type
直接latent type 推断型。避免到时候越来越乱，一开始弄好然后就debug。
可以尝试实现这两种tutor看看运行正常不正常：
有一种tutor是目的是最大化learner在后期eval的泛化效果。这个tutor允许learner的在tutor干预过程中的失败率的存在。
另一种是保证learner的在tutor干预过程中的失败率比较低，但是尽可能维持住最大化learner在后期eval的泛化效果。
需要记录的有：teach干预过程的平均死亡次数，超时次数比例，平均confirm次数(成功案例的)。还有eval过程中的死亡率、成功率、超时次数比例。
当 ΔTeachSuccess 很高而 ΔEvalPenalty 很低时，允许 hint
当 ΔTeachSuccess 一般，但 ΔEvalPenalty 很高时，禁止过度 hint
用一个 dependence / over-help penalty 显式控制

13. 评测口径还差一个“单位”
平均 confirm 次数：是只对成功样本算
teach 阶段死亡次数：按 query 平均
eval 成功率：按 query 成功率
timeout：是 confirm 超限才算 timeout

14. 我觉得还差一个最重要的设计决策：teach 阶段到底允许 learner 学到“完整答案”吗
最佳的情况是避免time out同时保证eval，比如放入一个球，看看情况，然后放入两个。这里就设计一个gate之类的，就是什么时候是尽可能帮助完成，什么时候是使得eval成绩更好，就相当于有两种类型tutor。这里的gate主要是由于：可能我如果不放球，虽然他会失败，但是他可能学更多？当然这个也要看实验。

15. confirm 错误后的反馈内容
可以设置config
基础是：只知道错。
另一种是：知道哪些位置错。
但是要求learner能根据错误，或者说知道自己这个是错的，能更新自己的模型。

16. select 集合中多个球的 risk 结算
如果一个 select 里拿了 5 个球，其中 2 个危险：warning 是只告诉“这组有危险”。通过使用Pragmatic Inference (Rational Speech Acts)的Multi-Intent Inference (Cardinality & Sets)来建模。

17. tutor 何时能发 hint
只有在confirm之后，tutor才会评估是否需要hint。也只会在这个阶段发hint，tutor选择发不发之后，才到learner的下一轮。

18. Candidate pool color distribution — what ratio of target-palette vs random colors?：
那个grammar里面出现的颜色，这些颜色里面随机。注意，一个learner和tutor的整个流程过程（包括预先训练和eval）用一个文件里面的grammar，不跨grammar来评估。






1. 目前测试的内容，对应的query问题是如何产生的，是直接从txt里面获取，还是通过生成器生成？生成的题目请生成10个放到txt里面让我看看。看看是不是过于简单，导致Compressed = exact和Shadow divergence = 0问题

2. 之前测试时候的n sup，obs轮次，teach轮次，eval轮次各是多少？

3. hint的机制是否成功预估了可能的time out，以及hint机制提高了避免time out的能力














