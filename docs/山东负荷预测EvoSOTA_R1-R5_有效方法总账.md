# 山东负荷预测自动优化（R1→R5）经过验证的方法与知识总账

> 这是从 R1 到 R5 五轮自动优化中**经过验证的有价值的方法和知识（knowledge）**。凡是下面记为"有效"的方法，都不仅在"当月 30 天快考"上成立，而且在 **full 回测——也就是横跨 5 个月（2025 年 9 月、12 月、2026 年 3 月、5-6 月，共 111 天）的跨月回测**上同样有效；只涨当月、跨月不涨的方法一律判为"背题"，记为无效，不进入有效清单。

## 怎么读这份总账（三条验证纪律）

1. **双考制**：每改一步，先考"当月 30 天"（快，用来筛选）；只要刷新高就加考"跨月 111 天"（慢，用来裁决）。当月涨、跨月跌 = 判定为"背题"，作废回滚。五轮里大约一半"看似有效"的改进是被这条纪律抓出来的。
2. **确定性复跑**：整套评测管线每次跑分完全一致（同一配置跑两次，分数逐位相同）。因此任何 0.005 级别的微小涨跌都是真实因果，不是运气，才敢做"一次只改一个钮"的精细实验。
3. **失败账本**：每个死掉的尝试都被记下"为什么死"，后面的轮次禁止重犯。下面很多方法是踩着前一轮的失败才成立的。
- 代码位置：双考制落在各岛 tools/record_score.sh + tools/evaluator.py（protected，禁止修改）与 config 的 eval_mode_per_iter=medium / eval_mode_validate=full；镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/champion_carry_best_43f7186/tools/。知识结晶：各岛 final_report 的 dual-eval 记录与 idea_library.md 失败账本。

## 先解释后面反复出现的词

- **校正（预测后修正）**：模型预测完之后，用"最近几天预报 vs 实际"的偏差去修正未来预报。好比气象台知道"我最近总把午后报高 2%"，下次就主动调低。
- **多模型平均**：训 5-10 个"配方相同、随机种子不同"的模型取平均，单个模型的偶然手抖被互相抵消。
- **度日（HDD/CDD）**：偏离舒适温度（本项目实测拐点约 14℃）的度数。比 14℃ 冷 5 度记 5 个"取暖度日"（HDD），热 8 度记 8 个"制冷度日"（CDD）。它把"温度-用电"的 V 形关系翻译成模型能直接用的线性数字。
- **门控**：一个特征/规律只在它物理上成立的时段或条件下生效（例如节假日标志只在早 8 点前、晚 6 点后生效，因为白天用电被屋顶光伏盖住）。
- **相对误差 vs 绝对误差**：绝对误差=差几兆瓦；相对误差=差百分之几。考试（评分口径）用的是相对误差，而模型默认训练优化的是绝对误差——两者不一致会"练偏"。

---

## 第一阶段（R1-R3）：打地基——"预测什么"和"怎么学"

**1. 大扫除：删无关特征、删帮倒忙的特征（有效）**
- 做法：删掉风向（只影响发电侧、不影响用电侧，无物理通路）；删 6 个温度差衍生特征（"今天比昨天冷几度"之类的二次加工列，看着相关实则放大噪声）。
- 前后对照：删风向 +0.28（R1）；删温度差衍生 +0.44（R2）。但 R3 再删别的（冗余辐射列 −0.08、再删风向 −0.21）全败——删特征不是越多越好，这个集合已删到最优边界。
- 原理：树模型每棵树只随机看 80% 特征列，列越多，好列被"挤掉"概率越大。
- full 回测状态：进入冠军线，跨月有效。
- 服务器代码：committed_islands/round1/island_residual_ratio_target 的配置面（删除动作 git commit 2703bff "iter-33: Drop wind_direction_10m"）；现行各岛 config/model/lgb_targetday_medium_strict_safe.yaml 白名单均不含该列
- Git 仓库：岛历史 @ 2703bff（TR2 bundle 仓）；镜像快照见 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/champion_carry_best_43f7186/config/model/lgb_targetday_medium_strict_safe.yaml
- 知识结晶：committed_islands/round1/island_residual_ratio_target/.evosota/output/results/scores.jsonl（iter-33 行）+ final_report.md
```yaml
# 天气特征白名单（节选）——风向只影响发电侧、不影响用电侧，R1 移除后 +0.28
weather_cols: [t2m, ghi, …]   # wind_direction_10m 已不在列表中
```

**2. 把物理规律直接喂给模型（有效，但"喂法"有边界）**
- 做法：人工造度日特征（见词表）+ "比最近一周冷/热多少"的 7 天温度偏离 + 节假日类型标志；并对度日加"只许单调"约束（度日越多用电越高）。
- 前后对照：R3 一波主升浪（累计约 +0.4）；R4、R5 换数据后重加仍各涨 +0.08/+0.03。但 R5 给同一族特征加"全局"单调约束 −0.065——冷热效应只在特定时段成立，全局约束挡住了模型该做的分时段切分。规律要喂，不能喂太死。
- full 回测状态：度日族与温度偏离跨月有效（R4 温度偏离在 D3-D5 远期 +0.08~0.10、跨月 +0.12）。
- 服务器代码：islands-r5/island_leadtime_holiday/models/lgb_targetday_model.py:720（度日计算）；islands-r5/island_champion_carry/models/lgb_targetday_model.py:1149（单调约束）
- Git 仓库：leadtime_holiday 岛 master @ 3a85da8，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/leadtime_holiday_best_3a85da8/models/lgb_targetday_model.py；champion 岛 @ 43f7186
- 知识结晶：islands-r5/island_leadtime_holiday/.evosota/output/results/final_report.md（What Worked #3）+ scores.jsonl iter-2/4
```python
# 度日：把 V 形温度-用电关系翻译成线性数字（t_ref 实测拐点 14℃）
target_hdd = max(0.0, t_ref - target_daily_mean)   # 取暖度日
target_cdd = max(0.0, target_daily_mean - t_ref)   # 制冷度日
# 单调约束：度日越多用电越高（R3 +0.038；R5 证明不能加全局约束，要分时段）
```

**3. 训练目标与考试对齐（有效一次，过度对齐有害）**
- 做法：把"每次分裂选哪个好"和"何时停止训练"的衡量标准，从"差几兆瓦"（绝对误差）换成"差百分之几"（相对误差/MAPE），让训练和考试打分一致。
- 前后对照：R2 引入即进冠军线；R5 再试"更激进对齐"（给计分时段大幅加权、非计分时段降权）−0.10 ~ −2.43 惨败——一棵树服务所有时段，饿死非计分会连累计分时。
- full 回测状态：基础对齐跨月有效。
- 服务器代码：islands-r5/island_leadtime_holiday/models/lgb_targetday_model.py:15-24
- Git 仓库：master @ 3a85da8，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/leadtime_holiday_best_3a85da8/models/lgb_targetday_model.py
- 知识结晶：同岛 final_report.md（iter-15 行）+ scores.jsonl iter-15
```python
def _mape_style_eval(y_true, y_pred):
    # iter-15：早停判据换成与考试一致的「平均相对绝对误差」，
    # 让模型选择直接优化考试口径，而不是默认的绝对兆瓦误差
```

**4. 防过拟合三件套（有效）**
- 做法：过滤"增益太小的分裂"（收益不到门槛就不许分，视为记噪声）；叶权重 L1 稀疏化（把没信号的叶子压到正好 0）；叶输出向父节点平滑（叶子别太自信）。
- 前后对照：R3 累计 +0.08 量级；此后各轮沿用，无失败记录。
- full 回测状态：随冠军线跨月有效。
- 服务器代码：islands-r5/island_champion_carry/config/model/lgb_targetday_medium_strict_safe.yaml:11-21
- Git 仓库：champion 岛 master @ 43f7186（三件套源自 R3 冠军血脉 @ 19dbee7，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/r3_champion_lineage_best/config/model/lgb_targetday_medium_strict_safe.yaml）
- 知识结晶：committed_islands/round3/island_residual_ratio_target/.evosota/output/results/final_report.md（Regularization stack 表 iter-72/75/79/94）
```yaml
reg_alpha: 0.05          # L1 叶权重稀疏化：没信号的叶子压到正好 0
path_smoothing: 0.1      # 叶输出向父节点平滑：叶子别太自信
min_gain_to_split: 0.001 # 增益太小的分裂直接禁止：关死记硬背的通道
```

**5. 比例/加法混合校正（有效，至今仍在使用且未调到头）**
- 做法：校正偏差有两种算法——加法（"午后总高 200 兆瓦，减掉"）和比例（"午后总高 2%，除回去"）。R1 先试只比例（−0.37 惨败）、只残差（只预测比参照日多/少多少，也败），**两者按 0.7/0.3 混合才转正**；R2-R3 把混合比例做成"分时段、分提前天数"各不同，成为冠军核心并继承到 R5。
- 为什么有效：比例法在用电水平大幅波动/换月时稳（误差跟水平成正比），加法法在形状固定的光伏午间低谷时准；混合=各取所长。
- full 回测状态：跨月有效；R5 报告自己点名"按波段细分混合比例"仍是头号未饱和方向。
- 服务器代码：islands-r5/island_champion_carry/models/lgb_targetday_model.py:1213-1217、1316-1321
- Git 仓库：master @ 43f7186，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/champion_carry_best_43f7186/models/lgb_targetday_model.py
- 知识结晶：R1 hybrid blend 见 committed_islands/round1/…/scores.jsonl iter-11..14；分时段 alpha 见 round3 final_report（per-dayplus alpha overrides，最大机制族）
```python
ratio_inverse = ratio_pred * max(gap_load_same_slot, 1.0)  # 比例模型×参照=兆瓦
pred = alpha * pred_additive + (1 - alpha) * ratio_inverse # 两种校正按比例混合
# alpha 按 hour_bin / dayplus 细分：午间光伏低谷信加法、夜间水平波动信比例
```

## 第二阶段（R4）：把校正和平均做成科学

**6. 校正借力（部分池化，至今主力，跨月效果放大近百倍）**
- 做法：校正量按"时段×提前天数"分小格子估计；格子数据少就不可信，于是往"同一时段平均值"拉，预测越远拉得越多（D1 信自己 70%，D5 只信 40%）。
- 前后对照：当月仅 +0.002，跨月 +0.15。R5 又做"普查"：发现模板继承的收缩系数从没调过，午间调到 0.5 得跨月 +0.079（该轮最大单步）。
- full 回测状态：跨月有效，且是换月时最痛问题的对症解。
- 服务器代码：islands-r5/island_champion_carry/models/lgb_targetday_model.py:1287-1300；yaml calibration.pool_lambda
- Git 仓库：master @ 43f7186（方法首发 R4 champion 岛 iter-1，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/champion_carry_best_43f7186/…）
- 知识结晶：islands/island_champion_carry/.evosota/output/results/final_report.md §Change A（full +0.154 的来源）
```python
mean_hourbin = mean(residuals[同时段全部天数])     # 大数据先验
cell_mean    = mean(residuals[该时段×该提前天数]) # 小格子估计
bias = lam * cell_mean + (1 - lam) * mean_hourbin  # 越远/越少数据，越信先验
```

**7. 稳健平均（删最远再平均，有效）**
- 做法：5 个模型平均时删掉离大伙最远的一个再平均，只用在 D3 以后（远期才需要）。
- full 回测状态：随 R4/R5 冠军跨月有效。
- 服务器代码：islands-r5/island_champion_carry/models/lgb_targetday_model.py:1392-1406；yaml seed_bag.agg: trim1
- Git 仓库：master @ 43f7186（首发 R4 iter-8，镜像同上）
- 知识结晶：islands/island_champion_carry/.evosota/output/results/final_report.md §Change B
```python
if agg == "trim1":   # 只用于 D3-D5（远期才需要）
    row_med = nanmedian(preds_per_seed, axis=1)
    # 删掉与行中位数偏差最大的那个 seed，其余平均（无偏方差缩减）
```

**8. 让多模型"真正不一样"（无偏降方差三招，有效；"换强度"无效）**
- 做法：①种子 5→10 纯平均；②一半模型换成"随机选阈值"版（普通树选最优切分点，它随机选，成员才真正不相关）——全场最大单次跨月跳变 +0.61；③打开行抽样（发现配置里这个开关一直没生效，"考古"出来的胜利）。
- 前后对照：循环换 bin 数/列采样 −0.05（不增多样性，只让成员水平不齐）；10 个全换随机阈值 −0.11（夜间断面需要精确阈值）。只有"无偏地增加差异"才赚。
- full 回测状态：跨月有效。
- 服务器代码：islands/island_horizon_leadtime/config/model/lgb_targetday_medium_strict_safe.yaml（seed_bag/extra_trees/subsample 块）
- Git 仓库：horizon_leadtime 岛 master @ 98b9934，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/horizon_leadtime_best_98b9934/config/model/lgb_targetday_medium_strict_safe.yaml
- 知识结晶：islands/island_horizon_leadtime/.evosota/output/results/final_report.md（Key Changes #4/5/6 + 失败对照 iter-27/28/29）
```yaml
seed_bag:  count 10, 其中 5 个 extra_trees  # 随机阈值版→成员真正不相关
subsample: 0.9
bagging_freq: 1   # 行抽样开关：R4 发现它一直没生效，打开即涨（+0.025）
```

**9. 门控（R4 诞生，R5 最大杠杆的雏形，有效）**
- 做法：节假日标志只在早 8 点前、晚 6 点后生效。
- 前后对照：不加门控 −0.19，加了 +0.05；R5 不加门控版当月 +0.055 但跨月被否决。
- full 回测状态：门控版跨月有效。
- 服务器代码：islands-r5/island_leadtime_holiday/models/lgb_targetday_model.py:865-868（首发 R4 horizon 岛 iter-16）
- Git 仓库：master @ 3a85da8，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/leadtime_holiday_best_3a85da8/models/lgb_targetday_model.py
- 知识结晶：R4 horizon final_report iter-16（−0.19/+0.05 对照）+ R5 leadtime final_report iter-39（门控推广到 anchor 侧）
```python
is_demand_regime = hour_val < 8 or hour_val >= 18  # 早8前/晚6后：需求侧可见
is_holiday = 1 if (target_day in 节假日 and is_demand_regime) else 0
# 白天被光伏盖住，节假日效应只在非白天生效；不加门控 −0.19，加了 +0.05
```

**10. 双考制立为纪律（全场最值钱的"方法"）**
- R4 起"当月涨/跨月跌=作废"抓回 3 次，包括一次"当月新高"的诱惑。它不是模型方法，是判决方法，但五轮里保住的分数比任何单招都多。

## 第三阶段（R5）：最近的新东西

**11. 节假日全过程建模（本轮最大单项增益，+0.3 量级，有效）**
- 做法：把节假日从"一个开关"升级成完整物理过程：①大假/小长假分开（停工幅度差 3-10 倍）；②门控（继承 R4）；③假期有"渐停→假中最深→末日提前复工"的爬坡；④复工后 1-3 天用电冲高；⑤修复"被假期压低的历史日被误当结构变化"的隐蔽错误。
- 前后对照：不加门控版跨月被否；月末工业冲刺只在 D1 有效、远期得不偿失被弃。
- 为什么有效：山东是工业省，假期停工是比天气还大的波动（−3%~−32%），而一年没几个大假，样本极少，模型自己学不出完整过程。
- full 回测状态：跨月有效（final 94.82，当时全场最高）。
- 服务器代码：islands-r5/island_leadtime_holiday/models/lgb_targetday_model.py:91-126
- Git 仓库：master @ 3a85da8，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/leadtime_holiday_best_3a85da8/models/lgb_targetday_model.py
- 知识结晶：islands-r5/island_leadtime_holiday/.evosota/output/results/final_report.md（iter-1/39/40 三块，全场最大增益链）
```python
# 假期位置归一化：0=首日(渐停)、1=末日(提前复工)——停工是爬坡不是一刀切
# 复工标记：多日假结束后 1-3 天用电冲高（补单开工），单独做特征
```

**12. 已兑现误差反馈（最新纪录 95.15 的来源，跨月 +0.21，有效）**
- 做法：模型每天预报未来 5 天；过一两天其中 D1/D2 就"兑现"，能算真实误差。把最近 3 次已兑现误差按新鲜度加权（越近越信，每天打 75 折），从当前 D1/D2 预测里扣掉。
- 前后对照：12 种"更聪明估计器"（中位数、截尾、砍极端值、按信噪比开关）全输给"朴素加权平均"——因为大误差恰恰是"高偏差状态"（坏天气系统持续），正是要修的对象，不能删不能缩；D3-D5 不门控无效（被天气预报误差淹没）。
- 为什么有效：模型偏差是"状态相关"的——昨天偏，明天大概率还偏；这个先验只在 1-2 天内成立。
- full 回测状态：跨月有效，把历史纪录从 94.94 推到 95.15。
- 服务器代码：islands-r5/island_champion_carry/models/lgb_targetday_model.py:1419-1485；yaml:153-164
- Git 仓库：master @ 43f7186，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/champion_carry_best_43f7186/
- 知识结晶：islands-r5/island_champion_carry/.evosota/output/results/final_report.md（What Worked 4 连 + 37 失败轴清单）
```yaml
error_feedback:
  enabled: true
  n_origins: 3        # 只用最近 3 次已兑现误差（更远的是已翻转的偏差）
  max_dayplus: 2      # 只改 D1-D2（偏差的持续性只在 1-2 天内成立）
  recency_decay: 0.75 # 每旧一天打 75 折：越近越信
```

**13. 分时段校准工艺（白天/夜间分开治，有效）**
- 做法：白天误差来自辐射预报、随天数增大→用温和加法+多借力；夜间是稳定基荷→用比例法+敢校正。一个常数一个常数按波段拆开调。
- 前后对照：R5 的 VAE 方向岛靠它跨月 +0.10，且十个存活变更全是这类工艺、零特征新增。
- full 回测状态：跨月有效。
- 服务器代码：islands-r5/island_wildcard_capacity_routing/config/model/lgb_targetday_medium_strict_safe.yaml:152-160
- Git 仓库：capacity_routing 岛 master @ 8f0dcc6，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/capacity_routing_best_8f0dcc6/config/model/lgb_targetday_medium_strict_safe.yaml
- 知识结晶：同岛 final_report.md（iter-33/34 校准普查，含 6 点 bracket 数据）
```yaml
shrinkage_by_hour_bin:
  2: 0.5    # 午间校正砍半：模板 1.0 过度校正（跨月 +0.079，该轮最大单步）
  40: 1.25  # 夜间敢校正：基荷稳定、信噪比高（跨月 +0.012）
```

**14. 用"预报间分歧"当特征（新信号源，有效；叠加同类无效）**
- 做法：数值天气预报有多个成员（同一天的多个平行预报），"成员之间温度差多少"=天气不稳的程度=模型该保守的信号。+0.12，该轮最大单步。
- 前后对照：再叠"辐射分歧" −0.015（一种分歧度就饱和）；同比趋势 −0.079（单日形状差异里天气噪声是趋势的 5 倍）。
- full 回测状态：跨月有效。
- 服务器代码：islands-r5/island_wildcard_feature_budget/models/lgb_targetday_model.py:115-118
- Git 仓库：feature_budget 岛 master @ 900c272，镜像 https://github.com/qjly421/load-forecast-agent-plat/blob/main/code/feature_budget_best_900c272/models/lgb_targetday_model.py
- 知识结晶：同岛 final_report.md（iter-6 IDEA-029 +0.120；饱和对照 iter-7）
```python
self.enable_nwp_spread_features = bool(...)  # 数值预报多成员温度离散度
# 「成员之间差多少」=天气不稳的程度=模型该保守的信号；+0.12 该轮最大单步
```

## 被反复验证"此路不通"的清单（最可靠的负知识）

- 加模型复杂度（扩叶子、换目标函数、RevIN 归一化）：当月偶涨、跨月必跌或持平，37 次尝试 0 存活（R4 冠军岛）。
- 直接喂光伏预测数据当特征：两轮验证均无效（R4 −0.64、R5 控窗后 −0.14）。
- 过度对齐训练目标：−0.10 ~ −2.43。
- 叠同类特征（第二种分歧度、同比趋势）：饱和即败。
- 共同指向同一结论：**模型"学"的部分已饱和，未来增量在"校正层 + 人类先验 + 实时信息"**。

## 代码与证据位置

- **GitHub（本仓库）**：本文件即方法结晶；仓库 `https://github.com/qjly421/load-forecast-agent-plat`，路径 `docs/山东负荷预测EvoSOTA_R1-R5_有效方法总账.md`。
- **服务器原始证据（逐轮逐迭代记分、最终报告、git 历史）**：TR2 主机 `/data1/liujunqi/ljq/evosota-ljq/`，其中 `committed_islands/round1-3`（R1-R3）、`islands/`（R4）、`islands-r5/`（R5），各岛 `.evosota/output/results/` 下有 scores.jsonl 与 final_report.md。
- **图表**：发展脉络图与精度演化曲线已上传本仓库 figures/ 目录（shandong_evosota_r1_r5_lineage.png / .drawio / shandong_evosota_r1_r5_accuracy_curve.png）。
