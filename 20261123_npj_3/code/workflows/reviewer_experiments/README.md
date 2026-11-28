# 第一批审稿补充数值实验

分支：code/reviewer-experiments。复用原始 ManyBodyQWZ.py，不修改历史文件。

## 运行

在 code 目录、安装 numpy scipy matplotlib tqdm 的 Python 环境执行：

```powershell
python workflows/reviewer_experiments/run_interaction_controls.py --sizes 3 4 --time 10 --points 9
```

程序自行设置包搜索路径；结果保存到 results/reviewer_experiments。可用 --output 指定其他目录，避免不同时间扫描互相覆盖。

## 本次实际计算

- 两个不可分辨费米子，开边界，L=3 和 4，每元胞两个模式；扇区维数 153 和 496。
- U=0,0.25,...,2，共九点；固定 t=10。
- 三种条件：(u,gamma)=(-1.5,0.2)、(-2.5,0.2)、(-1.5,0)。topological / trivial 标签指厄米母体参数区，不是已计算的相互作用拓扑不变量。
- 初态为模式 1 和 2L+1 被占据的 Fock 态，即相邻 x 元胞、y=0 的 B 模式。
- 采用 exp(-iHt) 后归一化的有限时间纯态。不是默认长期稳态，也不是 MQC 有限 shots 模拟。
- 最大尺寸、拓扑增益参数下另计算 U=0、1 全谱，保存最大虚部本征态和残差。图中的密度、关联仍属于有限时间态。

## 指标定义

1. edge_weight：y=0 上的期望总粒子数除以 2。
2. rms_y：sqrt(sum_y y^2 n_y / 2)，单位元胞；这是相对边界的均方根距离，不是指数拟合的局域长度。
3. adjacent_pair：位于曼哈顿距离为 1 的不同元胞的无序模式对联合占据概率总和。两粒子情况下是概率。
4. connected_norm：C_ij=<n_i n_j>-<n_i><n_j> 的非对角 Frobenius 范数；热图显示完整矩阵，包括对角占据涨落。
5. spatial_entropy：按原费米模式顺序截取前 floor(L/2) 个 x 列形成空间 A 区，计算纯态 Schmidt 熵，使用自然对数。连续模式切分避免重排的费米符号；不将该量解释为超出交换效应的粒子纠缠。

## 产物

- interaction_controls.png/pdf：五项指标与三类对照，两个尺寸。
- spectrum_density_correlations.png/pdf：L=4、U=0/1 的全谱、有限时间密度和连通关联。
- scan.csv：全部 54 组指标。
- snapshot_L4_U*.npz：谱、态、基底、密度、关联和本征残差。
- metadata.json：参数、依赖版本及验证结果。

## 验证与解读

实际运行环境 Python 3.12；NumPy 2.3.5、SciPy 1.18.1、Matplotlib 3.11.2。
检查了无增益 Hamiltonian 厄米性、2x2 二激发维数、无相互作用二粒子谱与单粒子能量两两和匹配（最大误差约 9.35e-15）、乘积态零熵及已知二分态 ln(2) 熵、本征残差。
PNG 已人工视觉检查。

初步结果：有限时间边界权重随 U 非单调；平庸母体参数也有较强边界堆积；空间纠缠非单调，U=0 也可以非零。不能据这些图宣称多体拓扑保护、相变或相互作用必然增强纠缠。

尚未覆盖：长期/尺寸收敛、主导谱分支切换、沿边界局域尺度、手性反转、无序统计、粒子纠缠、MQC 精度/后选择/噪声。当前图是第一批计算诊断，不是全部审稿意见的最终证据。
