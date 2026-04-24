# IndoorPos

[English](README.md) | 简体中文

这是一个已经归档的 RSSI 室内定位实验仓库。

这个项目实际尝试过的方向很多：

- 面向 ESP32 / ESP32-S3 的超小模型
- 纯 RSSI 指纹定位基线
- 更强的传统机器学习定位方法
- 更大的 Torch 模型
- 伪轨迹 / 序列建模方法
- 面向服务器的多模型 benchmark 工具链

最后的结论是：仓库里沉淀出了不少有价值的实验和可复现脚本，但整体结果还没有强到值得继续长期投入，所以仓库保留为“实验记录 + 基准参考”，不再继续活跃开发。

## 仓库状态

已归档。

保留这个仓库的目的主要是：

- 保留实验过程和结果
- 方便别人复现实验
- 给后续类似 RSSI 定位项目提供参考

## 目录结构

```text
IndoorPos/
├── README.md
├── README.zh-CN.md
├── scripts/         # 训练、评估、数据集生成、benchmark 入口
├── docs/            # 实验说明、部署/服务器指南、HTML 可视化
├── requirements/    # 本地 / 服务器 / CUDA 依赖集合
├── tools/           # 辅助 shell / bat 启动脚本
├── archive/         # 原始 CSV 指纹数据集
├── training_dataset/
├── test_dataset/
├── training_dataset_fixed/
├── test_dataset_fixed/
└── runs/            # 生成的实验结果
```

## 最终最靠谱的方案

对于**可部署的纯 RSSI 绝对定位**，这个仓库最后最好的方法不是神经网络，而是：

- 脚本：
  `scripts/TrainRSSITabularEnsemble.py`
- 方法：
  `ExtraTreesRegressor + 楼栋/楼层门控`
- 最优候选：
  `extra_trees_est1200_mf0.5_leaf1_flatten_stat_top1`

最终本地纯 RSSI 测试集结果如下：

| 方法 | mean (m) | median (m) | p90 (m) | p95 (m) | rmse (m) |
|---|---:|---:|---:|---:|---:|
| 树模型冠军 | 13.580 | 10.319 | 27.549 | 35.058 | 17.923 |
| 纯 RSSI WKNN 基线 | 13.724 | 8.155 | 32.120 | 47.370 | 21.903 |

这组结果要这样理解：

- `mean` 只比 WKNN 好一点点
- 但长尾误差明显更好
- `p90`、`p95`、`rmse`、`max error` 都优于 WKNN
- `median` 反而还是 WKNN 更好

如果是医院导航这种更怕“偶发大漂移”的应用场景，这个树模型方案比 WKNN 更值得优先考虑，因为它的尾部风险更小。

对应结果文件：

- `runs/local_pure_rssi/tabular_final/metrics.json`
- `runs/local_pure_rssi/tabular_final/test_best_single_scatter.png`
- `runs/local_pure_rssi/tabular_final/test_predictions_best_single.csv`
- `runs/local_pure_rssi/knn_baseline_last_mean/metrics.json`

## 数据集说明

这个仓库同时使用了原始 CSV 指纹数据和若干衍生出来的序列化数据集。

### 1. 原始 CSV

位于 `archive/`：

- `archive/TrainingData.csv`
- `archive/ValidationData.csv`

它们是整个仓库里所有实验的基础 Wi-Fi 指纹表。

### 2. 非 fixed 序列数据集

主要实验使用的是：

- `training_dataset/`
- `test_dataset/`

根据 metadata，这两套数据的关键属性是：

- 来源 CSV：
  `archive/TrainingData.csv` 和 `archive/ValidationData.csv`
- 序列长度：
  `5`
- 生成模式：
  `endpoint_path`
- 选择的 AP 数量：
  `128`
- 训练集：
  `15949`
- 验证集：
  `3988`
- 测试集：
  `1111`
- 数据增强：
  启用了 RSSI 噪声，标准差为 `2.0 dBm`

### 3. fixed 序列数据集

仓库里还有：

- `training_dataset_fixed/`
- `test_dataset_fixed/`

它们用于部分 fixed-dataset 和 hybrid 相关实验。

### 4. 数据集重建

主生成脚本：

- `scripts/DatasetProc.py`

可视化脚本：

- `scripts/visualize_dataset.py`

示例：

```bash
python3 scripts/DatasetProc.py \
  --input-csv archive/TrainingData.csv \
  --output-dir training_dataset \
  --seq-len 5
```

如果需要更完整的服务器侧数据重建与训练流程，可以看：

- `docs/SERVER_TRAINING_GUIDE.md`
- `docs/DEPLOY_SERVER.md`

## 尝试过的算法

这里把仓库里真正写过、跑过的算法路线都列出来。

### 1. 面向 ESP32 的 tiny 模型

脚本：

- `scripts/TrainTinyESP32Model.py`

支持的结构：

- 深度可分离卷积网络 `dscnn`
- GRU `gru`
- TCN `tcn`

目标：

- 尽量小
- 尽量适合 ESP32-S3 侧部署

相关脚本：

- `scripts/EvaluateTinyConsensus.py`

这个脚本会做多次带扰动推理，再做共识聚类，看 tiny 模型能不能靠多次投票改善稳定性。

### 2. 纯 RSSI 的 WKNN 基线

脚本：

- `scripts/TrainRSSIKNNModel.py`

方法：

- RSSI 指纹特征
- KNN / WKNN 检索
- 支持加权 / 非加权
- 支持 group-aware 查找
- 支持可选时间滤波

这是整个仓库里最强、也最值得保留的基线之一。

### 3. 纯 RSSI 高级检索集成

脚本：

- `scripts/TrainAdvancedRSSIEnsemble.py`

方法：

- 多种 RSSI 特征视图
- 学习式楼栋/楼层分类器
- group-aware KNN
- 多种邻居聚合方式：
  `idw`、`idw2`、`kernel`、`softmax`、`trimmed_idw`、`median`、`lle`
- 基于验证集的贪心集成

这是纯 RSSI 检索路线里最完整的一套实现。

### 4. 纯 RSSI 树模型集成

脚本：

- `scripts/TrainRSSITabularEnsemble.py`

方法：

- 用 RF / ExtraTrees 做楼栋楼层分类
- 用全局或局部分组回归器预测坐标
- 主要回归器：
  `ExtraTreesRegressor`、`RandomForestRegressor`
- 验证集选优
- 可选候选集成

这就是仓库最后推荐的纯 RSSI 主方案。

### 5. 高精度传统模型

脚本：

- `scripts/TrainHighAccuracyModel.py`

方法：

- 准确率优先
- 楼栋楼层分类 + group-aware KNN

这条路在早期很有参考价值，但不是最终最优的纯 RSSI 方案。

相关说明：

- `docs/HIGH_ACCURACY_GUIDE.md`

### 6. 高精度 Torch 序列模型

脚本：

- `scripts/TrainHighAccuracyTorchModel.py`

方法：

- 序列编码
- GRU 时间建模
- embedding 学习
- kNN refinement

这个脚本偏向服务器侧高精度探索。

### 7. 纯 RSSI Torch 大模型

脚本：

- `scripts/TrainRSSIOnlyHighAccuracyTorch.py`

方法：

- 纯 RSSI 序列输入
- GRU 时序建模
- embedding + 坐标头
- 可选 kNN refinement

这个方向在 Apple Silicon 的 MPS 上也实际跑过，但最终仍然没有超过树模型冠军。

### 8. 绝对坐标 RSSI-only 神经网络

脚本：

- `scripts/TrainAbsoluteRSSIOnly.py`

方法：

- 纯 RSSI 输入
- 直接回归绝对坐标
- 辅助 grid 分类头

这条路的好处是不依赖轨迹递推，但最终效果没有成为仓库最优。

### 9. 轻量级序列模型 zoo

脚本：

- `scripts/TrainLightweightSchemeZoo.py`

尝试过的家族：

- `set_tcn`
- `cnn_tcn`
- `pure_tcn`

目标：

- 比 tiny 更强
- 但比大模型更轻

### 10. Hybrid CNN + TCN

脚本：

- `scripts/TrainHybridModel.py`

目标：

- 在 fixed dataset 上尝试 hybrid 时序建模

### 11. 文章风格的轨迹模型

相关脚本：

- `scripts/TrainArticleTrajectoryModel.py`
- `scripts/ArticlePureTCNModel.py`
- `scripts/TrainAndVisualizeArticlePureTCN.py`
- `docs/article_model_visualization.html`

方法：

- Top-K AP token / 展平
- TCN 时序建模
- 预测位移增量
- 辅助 grid 分类
- 可选后处理

这些模型在某些短窗评估里能得到很低误差，但它们不是最终推荐的纯 RSSI 绝对定位方法，因为部署假设和评测设定没有那么干净。

## 统一 benchmark 入口

### 纯 RSSI benchmark

脚本：

- `scripts/RunPureRSSIBenchmarks.py`

这是复现纯 RSSI 最终对比的最佳入口。

示例：

```bash
python3 scripts/RunPureRSSIBenchmarks.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-root runs/pure_rssi_bench
```

### 服务器多模型 benchmark

脚本：

- `scripts/RunServerBenchmarks.py`

配套脚本：

- `tools/run_full_benchmark_server.sh`
- `tools/setup_server_env.sh`
- `docs/SERVER_TRAINING_GUIDE.md`

适合高性能服务器上同时跑多种模型做总对比。

## 如何复现

### 1. 安装依赖

如果只复现纯 RSSI 的 CPU benchmark：

```bash
python3 -m pip install -r requirements/requirements-pure-rssi-bench.txt
```

如果还要跑 Torch 实验：

- `requirements/requirements-torch-cu118.txt`
- `requirements/requirements-torch-cu126.txt`
- 或者本地 Apple Silicon + MPS 可用的 PyTorch 环境

### 2. 复现 WKNN 基线

```bash
python3 scripts/TrainRSSIKNNModel.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/knn_baseline_last_mean \
  --feature-set last_mean \
  --k-candidates 1,3,5,7,9,11,15,21 \
  --weighted
```

### 3. 复现最终纯 RSSI 冠军

```bash
python3 scripts/TrainRSSITabularEnsemble.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/tabular_final \
  --feature-sets flatten_stat \
  --group-classifier-candidates rf:700:sqrt:stat_stack,extra_trees:900:sqrt:stat_stack,extra_trees:900:sqrt:quantile_stack \
  --regressor-candidates "extra_trees:1200:0.5:1:flatten_stat:global,extra_trees:1200:0.5:1:flatten_stat:top1,extra_trees:1500:0.35:1:flatten_stat:top1" \
  --ensemble-max-candidates 3 \
  --ensemble-max-steps 3 \
  --n-jobs -1
```

### 4. 复现 Apple Silicon / MPS 对比实验

```bash
python3 scripts/TrainRSSIOnlyHighAccuracyTorch.py \
  --train-dir training_dataset \
  --test-dir test_dataset \
  --output-dir runs/local_pure_rssi/rssi_torch_mps \
  --device mps \
  --epochs 100 \
  --batch-size 384 \
  --patience 12 \
  --candidates 192:192:2:256:0.15,256:256:2:320:0.18,320:320:2:384:0.20 \
  --num-workers 0
```

## 为什么归档

这个仓库不是没有成果，而是成果没有好到支撑继续做成一个“强成品”。

更准确地说：

- 纯 RSSI 方法确实有提升
- 但提升幅度没有大到让人满意
- 部分轨迹模型在某些评测设定里很好看，但离真实部署还有距离
- 项目沉淀了很多实验分支，但没有形成一个让人真正满意的最终答案

所以最诚实的收尾方式就是：

- 把代码保留下来
- 把复现路径写清楚
- 把结论写清楚
- 不再把它包装成一个已经做成的优秀系统
