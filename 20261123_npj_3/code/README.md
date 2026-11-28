# Paper3：关联手性皮肤态代码

本目录用于论文 Hybrid Chiral Skin Correlated States In Monitored Quantum Circuits 的修订。2026-11-28 从 QuantumPhysics1 复制并整理。

## 组织原则

全部 21 个原始 Python / Notebook 文件逐字节保留，包括 Notebook 输出、历史函数和不同版本；本轮只调整部分文件位置，未修改物理模型、算法或默认参数。原路径和新路径见 [source_manifest.json](source_manifest.json)。这是功能分类整理，不代表所有程序已经成功运行。

核心模块和主 Notebook 保留原导入路径；确认用途的辅助脚本归入 workflows；不能确认论文用途或版本关系的代码归入 unclassified，不能因名称相似就删除或合并。

## 目录与入口

```text
code/
  MeasureNonhermitianSkin/
    Experimental/
      ManyBodyQWZ.py                 # 2D 固定粒子数模型及数值实验
      ManyBodyCSTI.py                # 3D 模型及数值实验
      Circuit_ManyBody.py            # MQC 电路实现
    MBPumpLocalization/
      SingleExcitation.ipynb         # 论文主计算、数据读取与绘图
  ComputationalPhysics/Package/
    BitOperation.py                 # 费米子基底位运算
  workflows/
    vqe/                            # 7 个训练/运行/展示脚本及 Spectrum.ipynb
    circuit_optimization/           # opti_circuits.py 与 opti_circuits2.py
    circuit_examples/               # Circuits.py
  unclassified/
    circuit_variants/desktop/       # 桌面同步副本
    circuit_variants/mbpump/        # MBPumpLocalization 电路变体
    exploratory_notebooks/         # ManyBodyQWZ 与 NumericalChiral_22
    platform_examples/             # LearnQuanfu.py
```

### 核心功能

| 文件 | 已识别功能 |
| --- | --- |
| ManyBodyQWZ.py | ParticleConservedSystem / ParticleConservedSystem_Coo；固定粒子数基底、费米符号、QWZ 跃迁、边界非厄米项、胞内/胞间相互作用、占据数；Comp、FockSkin、CalPopu、DifInter 等数值实验 |
| ManyBodyCSTI.py | RealCSTI_ham、ParticleConservedSystem_CSTI / CSTI_Coo；3D 模型及 CompHam、CompTwoExcit 对照 |
| Circuit_ManyBody.py | Circuit/Circuit2/Circuit3；CircuitSystem、CircuitSystem2、CircuitSystem_Full、CircuitSystem_MPS；厄米门、非厄米操作、相互作用与演化 |
| SingleExcitation.ipynb | 名称虽含 Single，实际涵盖二/三激发、3D 及 MQC；包含参数计算、数据加载、关联和图形导出 |
| workflows/vqe | 保留所有 TrainVQE、RunCodeManyBody 的变体；包含优化、单/多粒子试验及 HDF5 展示，不指定某个后缀版本为最终版本 |
| workflows/circuit_optimization | 分层变分线路构造、目标态拟合与优化；两个版本分别保留 |
| workflows/circuit_examples | get_wavefunction 系列，含非厄米和单次测量电路示例 |

主 Notebook 的可检索单元位置见 [NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md)。其中包括 Pic_TwoExcitation.pdf、MQC_Comp.pdf、ThreeExcitation_Popu_Interaction.pdf、SI_3DCSTISpetrum2.pdf 等导出位置；同名图可能有多个版本，需结合参数和论文核对。

## 待确认代码

- circuit_variants/desktop：与核心 Circuit_ManyBody.py 有实际内容差异，不当作完全重复文件删除。
- circuit_variants/mbpump：较短的电路实现，保留 CircuitSystem 及 fun1 等功能；主论文 Notebook 当前导入的是 Experimental 下的版本。
- exploratory_notebooks：模型探索与数值试验，尚未确认各单元与最终论文图的关系。保留所有单元和输出。
- platform_examples/LearnQuanfu.py：Quafu / pyquafu 平台探索，含顶层执行代码；使用前核对平台 API 和账号配置。
- VQE 同一文件可能重复定义函数，例如 VQE23；本轮保留原覆盖行为，不擅自合并。

## 环境与运行方式

建议使用原项目已验证的 Python 环境。源码混用了旧式 Qiskit（qiskit.Aer、execute、qiskit.opflow）和 qiskit_aer，尚没有一套经过验证的统一依赖版本；直接安装最新版不保证兼容。

常见依赖：numpy、scipy、matplotlib、tqdm、h5py、numba、plotly、Jupyter、Qiskit/Aer。Quafu 示例另需对应平台 SDK。没有将未经验证的版本写成锁定依赖。

在 PowerShell 中将 code 加入 Python 模块搜索路径，然后从 Notebook 所在目录启动 Jupyter：

```powershell
Set-Location E:\texstudiofile\paper3\20261123_npj_3\code
$paperCodeRoot = (Get-Location).Path
$env:PYTHONPATH = "$paperCodeRoot;$env:PYTHONPATH"
Set-Location .\MeasureNonhermitianSkin\MBPumpLocalization
python -m jupyter lab SingleExcitation.ipynb
```

Notebook 按单元选择执行，先检查参数和数据依赖。它包含历史实验及重复输出文件名，不建议直接 Run All。运行某个 workflows 脚本时，在设置上述 PYTHONPATH 后切换到该脚本目录，以使相对数据路径按其所在目录解析。

## 数据与输出

本次仅复制代码，外部 .npz / .h5 和 DataPicture 图片未复制。源数据仍位于 E:\PythonProject\QuantumPhysics1\MeasureNonhermitianSkin 下的原目录。绘图单元可能先读取既有文件，需复制对应数据或执行其生成单元；导出前创建 DataPicture 等目标目录。

移动后的 workflows / unclassified 文件仍使用原相对文件名，需要将所需数据放到实际工作目录。历史目录位置可查 source_manifest.json。不要把缺失数据误判为此次整理丢失算法。

## 验证和后续维护

在 code 目录执行：

```powershell
.\verify_preservation.ps1
```

该检查验证 21 个原始文件与整理时 SHA-256 完全一致，并验证 Notebook JSON 格式。它不导入科研模块、不触发训练或仿真。后续主动修改源码后，哈希检查失败是预期现象，届时应通过 Git 差异审查并更新基准。

本轮已通过文件保留与 Notebook 格式检查；尚未验证依赖安装、数值复现、运行耗时或全部脚本的端到端运行。修改物理计算前，应建立小系统基准，并明确主文图对应的单元、参数及数据版本。

原始复制说明保留于 README.txt，描述的是整理前快照；当前目录布局以本 README 为准。
