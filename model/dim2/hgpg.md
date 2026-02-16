# 3. Methodology

## 3.1. Overview
To address the challenges of vascular fragmentation and low contrast in X-ray angiography, we propose the **Hierarchical Geometric Prior Guidance (HGPG)** framework. The core philosophy is to transition from simple data-driven learning to **geometry-aware feature modulation**. By integrating a **FrangiSpectralFilter** for multi-scale geometric probing, a **Gated Spatial Modulation (GSM)** mechanism, and a topology-preserving loss function (**clDice**), the model effectively redistributes feature energy to enhance structural continuity across the entire coronary tree.

---

## 3.2. Frangi-based Geometric Probing
The system begins with the **FrangiSpectralFilter**, a specialized module designed to extract the **Integrated Geometric Prior (IGP)**. This module explicitly encodes the second-order local structure of the image intensity field, providing a robust structural initialization.

### 3.2.1. Parameter Optimization for Angiographic Data
To accommodate the high variation in vessel diameters within our dataset—ranging from large proximal trunks to delicate distal branches—we meticulously configured the filter's hyperparameters. We set the scale-space parameters to **$\sigma = \{1, 3, 5, 7, 9\}$**, ensuring a comprehensive capture of vascular structures across different spatial frequencies. The sensitivity parameters are defined as **$\beta = 1.0$** and **$C = 10.0$**, which are optimized to distinguish tubular coronary arteries from background noise and blob-like artifacts while maintaining a sharp response at vessel bifurcations.

### 3.2.2. Scale-Space Hessian Analysis
For a given input $I$, the module computes the Hessian matrix $\mathbf{H}$ at each scale $\sigma$. To ensure computational efficiency, the Hessian kernels are pre-computed and registered as structural buffers:
$$\mathbf{H}(I, \sigma) = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$$
where $D_{xx}, D_{xy}, D_{yy}$ are obtained via 2D convolutions with scale-normalized kernels $\sigma^2 \cdot \frac{\partial^2 G_\sigma}{\partial x_i \partial x_j}$.

### 3.2.3. Eigen-decomposition and Vesselness
For each scale, we perform eigen-analysis to derive eigenvalues $|\lambda_1| \leq |\lambda_2|$. The tubularity is quantified by the vesselness function $\mathcal{V}(\sigma)$:
$$\mathcal{V}(\sigma) = \exp\left(-\frac{\mathcal{R}_B^2}{2\beta^2}\right) \left(1 - \exp\left(-\frac{\mathcal{S}^2}{2c^2}\right)\right)$$
where $\mathcal{R}_B = \lambda_1 / \lambda_2$ is the blobness measure and $\mathcal{S} = \sqrt{\lambda_1^2 + \lambda_2^2}$ represents the structuredness. A foreground constraint $\lambda_2 < 0$ is applied to selectively highlight bright vascular structures. 



### 3.2.4. Integrated Geometric Prior (IGP)
The final **Integrated Geometric Prior** $\mathcal{P}_{geom}$ is derived through a max-pooling operation across the scale-space:
$$\mathcal{P}_{geom} = \max_{\sigma \in \{1, 3, 5, 7, 9\}} \mathcal{V}(\sigma)$$

---

## 3.3. Hierarchical Geometric Prior Guidance (HGPG)
The IGP is propagated through the MedFormer encoder via a three-tiered hierarchical strategy to maintain structural consistency from pixel-level inputs to deep semantic representations.

### 3.3.1. Tier 1: Early-Stage Multi-modal Fusion (Level 0/1)
At the highest resolution, the raw image $I$ is concatenated with $\mathcal{P}_{geom}$ to form a multi-modal input for the initial convolution layer ($inc$). This tier acts as a **geometric anchor**, prioritizing tubular gradients over planar noise before any spatial information is lost through downsampling.

### 3.3.2. Tier 2: Mid-Level Structural Modulation (Level 2)
At $1/4$ resolution, we implement the first **Gated Spatial Modulation (GSM)**. The IGP is downsampled via $2 \times 2$ Max-Pooling to match the feature dimensions. This tier serves as a **topology stabilizer**, amplifying the response of bifurcations and mid-sized vessel segments.

### 3.3.3. Tier 3: High-Level Semantic Modulation (Level 3)
At the deepest semantic level ($1/16$ resolution), $4 \times 4$ Max-Pooling is applied to the IGP. This tier acts as a **semantic refiner**, ensuring that the global vascular tree's continuity is reinforced within the bottleneck, providing a noise-free foundation for feature reconstruction.

---

## 3.4. Gated Spatial Modulation (GSM) Logic
Unlike additive fusion, the GSM module acts as a **feature purifier**. For any level $l$, a spatial-channel attention gate $\mathcal{G}_l$ is derived:
$$\mathcal{G}_l = \sigma(\text{BN}(\text{Conv}_{1\times1}(\text{MaxPool}(\mathcal{P}_{geom}, \downarrow_l)))) \cdot s_l$$
where $s_l$ is a **learnable scaling factor**. The features $F_l$ are refined via a **residual gating mechanism**:
$$\hat{F}_l = F_l \odot (1 + \mathcal{G}_l)$$
This formulation ensures that the network selectively enhances features in regions with high vascular probability while maintaining stable gradient flow.



---

## 3.5. Topology-Preserving Objective: clDice
Standard regional losses like Dice are biased toward thick proximal vessels. We incorporate the **clDice (Centerline Dice)** loss to optimize the topological overlap between the skeletons of the prediction $S_{pred}$ and the ground truth $G_{gt}$. The total objective is:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Dice} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{clDice}$$
By aligning the network's output with the **centerline skeleton** provided by the HGPG module, $\mathcal{L}_{clDice}$ penalizes structural disconnections and encourages global connectivity.



# 3. Methodology - Network Architecture

## 3.1. Proposed Architecture: MedFormer with Gated-HGPG
The proposed architecture integrates the **FrangiSpectralFilter** and **Hierarchical Geometric Prior Guidance (HGPG)** into the MedFormer backbone. The framework transition from simple concatenation to **Gated Spatial Modulation (GSM)** ensures precise feature purification.

---

### [图 3.1: MedFormer-HGPG 网络架构总览]

#### A. 输入层与 Tier 1 引导 (Resolution: 1x)
* **Input Image Branch**: 原始 X 射线造影图像。
* **Frangi Branch**: 通过 **FrangiSpectralFilter** (sigmas=1,3,5,7,9, beta=1, c=10) 提取的 **IGP** (Integrated Geometric Prior)。
* **Fusion Module (Tier 1)**: 将两者进行 **Concatenation**。
* **Encoder Entry**: 进入 `Conv Stem` 进行初步特征提取。

#### B. 编码器与层级门控引导 (Encoder Stages)
* **Stage 1 (Resolution: 4x)**:
    * **Bidirectional Transformer Block**: 提取 Transformer 特征。
    * **GSM Module (Tier 2)**: 
        * **Input 1**: Stage 1 特征流。
        * **Input 2**: MaxPool2d(IGP, 2)。
        * **Mechanism**: 执行门控空间调制 $\hat{F} = F \odot (1 + G)$。
* **Stage 2 (Resolution: 8x)**:
    * **Down-sampling**: 降采样特征。
    * **Bidirectional Transformer Block**: 特征增强。
    * **GSM Module (Tier 3)**: 
        * **Input 1**: Stage 2 特征流。
        * **Input 2**: MaxPool2d(IGP, 4)。
        * **Purpose**: 提纯瓶颈层语义，确保全局拓扑一致性。
* **Stage 3 (Resolution: 16x)**:
    * **Down-sampling**: 进入最深层（Bottleneck）。
    * **Multi-Scale Fusion**: 与各层级特征进行融合。



#### C. 解码器与拓扑约束 (Decoder Path)
* **Up-sampling & Skip Connections**: 标准 MedFormer 路径。
* **Segmentation Map Output**: 生成最终血管分割图。
* **Loss Supervision**:
    * **Regional Loss**: Dice Loss + CE Loss。
    * **Topological Loss**: **clDice Loss**（利用 IGP 辅助提取的骨架进行拓扑连通性约束）。

---

## 3.2. 修改点摘要 (Key Modification Summary)
| 模块 | 修改内容 | 物理意义 |
| :--- | :--- | :--- |
| **Input Tier** | 加入 FrangiSpectralFilter 分支 | 引入手工几何先验，辅助弱信号感知 |
| **Encoder Tiers** | 插入 GSM (Gated Spatial Modulation) | 从“信号叠加”转向“能量重分配”，抑制噪声 |
| **Resolution Adaptation** | 引入 $2\times$ 与 $4\times$ Max-Pooling | 确保几何先验在降采样过程中保留最强血管响应 |
| **Loss Function** | 加入 clDice (Centerline Dice) | 强制模型关注拓扑连通性，减少血管断裂 |