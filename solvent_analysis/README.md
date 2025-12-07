# 溶剂可及性分析工具包

模块化重构后的溶剂可及性分析工具，基于水分子接近度评估蛋白质残基的溶剂可及性。

## 项目结构

```
solvent_analysis/
├── __init__.py              # 包导出
├── __main__.py              # 模块入口点
├── README.md                # 本文档
├── core/                    # 核心数据模型和接口
│   ├── __init__.py
│   ├── data_models.py       # 数据类定义
│   ├── distance_calculator.py # 距离计算接口
│   └── accessibility_evaluator.py # 可及性评估接口
├── io/                      # 输入输出模块
│   ├── __init__.py
│   ├── pdb_loader.py        # PDB文件加载
│   ├── csv_writer.py        # CSV文件写入
│   └── result_formatter.py  # 结果格式化
├── algorithms/              # 算法实现
│   ├── __init__.py
│   ├── centroid_method.py   # 质心法
│   ├── peratom_method.py    # 原子级方法
│   ├── freesasa_wrapper.py  # FreeSASA包装器
│   └── method_factory.py    # 方法工厂
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── progress.py          # 进度条
│   ├── logger.py            # 日志工具
│   └── validation.py        # 验证工具
└── cli/                     # 命令行接口
    ├── __init__.py
    └── main.py              # 命令行主程序
```

## 核心概念

### 数据模型

- `ResidueInfo`: 残基信息（链、编号、名称、质心坐标）
- `WaterInfo`: 水分子信息（坐标、名称）
- `AccessibilityResult`: 可及性分析结果
- `AnalysisConfig`: 分析配置参数
- `MethodType`: 分析方法枚举（centroid/peratom）

### 设计模式

1. **策略模式**: 不同的距离计算和评估策略
2. **工厂模式**: 方法创建工厂
3. **依赖注入**: 通过构造函数注入依赖
4. **单一职责**: 每个模块/类只负责一个功能

## 使用方法

### 命令行接口（向后兼容）

```bash
# 使用原子级方法（默认）
python solvent_accessibility_new.py \
  --wet-pdb ./pdb/SUMO1_water.pdb \
  --dry-pdb ./pdb/SUMO1.pdb \
  --method peratom \
  --verbose

# 使用质心法
python solvent_accessibility_new.py \
  --wet-pdb ./pdb/SUMO1_water.pdb \
  --dry-pdb ./pdb/SUMO1.pdb \
  --method centroid \
  --verbose

# 不进行FreeSASA对比
python solvent_accessibility_new.py \
  --wet-pdb ./pdb/SUMO1_water.pdb \
  --dry-pdb ./pdb/SUMO1.pdb \
  --no-comparison
```

### Python API

```python
from solvent_analysis import AnalysisConfig, MethodType
from solvent_analysis.io import PDBLoader
from solvent_analysis.algorithms import MethodFactory

# 加载PDB文件
loader = PDBLoader()
residues, waters, structure = loader.load("protein.pdb")

# 创建配置
config = AnalysisConfig(
    threshold=3.5,
    radius=5.0,
    fraction_threshold=0.20,
    min_hits=1
)

# 创建分析方法
method = MethodFactory.create_method(MethodType.PERATOM, config)

# 执行分析
results = method.analyze(residues, waters, structure)

# 处理结果
for result in results:
    print(f"{result.residue.chain}{result.residue.resnum}: {result.accessible}")
```

## 模块说明

### core/ 核心模块

- **data_models.py**: 定义所有数据类，使用dataclass和类型提示
- **distance_calculator.py**: 距离计算抽象接口和具体实现
- **accessibility_evaluator.py**: 可及性评估抽象接口和具体实现

### io/ 输入输出模块

- **pdb_loader.py**: PDB文件解析，提取残基和水分子信息
- **csv_writer.py**: CSV文件写入，支持多种输出格式
- **result_formatter.py**: 结果格式化和对比表格生成

### algorithms/ 算法模块

- **centroid_method.py**: 质心法实现（基于残基质心）
- **peratom_method.py**: 原子级方法实现（基于每个原子）
- **freesasa_wrapper.py**: FreeSASA计算包装器
- **method_factory.py**: 方法创建工厂

### utils/ 工具模块

- **progress.py**: 进度条显示
- **logger.py**: 日志记录配置
- **validation.py**: 输入验证工具

### cli/ 命令行接口

- **main.py**: 命令行参数解析和主程序流程

## 性能优化

1. **分块计算**: 大矩阵计算分块进行，减少内存使用
2. **KDTree缓存**: 重用空间索引，避免重复构建
3. **向量化操作**: 使用NumPy向量化计算
4. **惰性计算**: 只在需要时计算原子级距离

## 扩展性

### 添加新的距离计算方法

1. 继承 `DistanceCalculator` 抽象类
2. 实现 `compute_min_distances` 和 `count_waters_within_radius` 方法
3. 在方法工厂中注册

### 添加新的可及性评估规则

1. 继承 `AccessibilityEvaluator` 抽象类
2. 实现 `evaluate` 方法
3. 在方法工厂中注册

### 添加新的输出格式

1. 在 `io/` 模块中添加新的写入器
2. 实现相应的格式化方法
3. 更新命令行接口支持

## 测试

运行测试脚本：

```bash
python test_refactored.py
```

运行完整功能测试：

```bash
python solvent_accessibility_new.py --wet-pdb ./pdb/SUMO1_water.pdb --dry-pdb ./pdb/SUMO1.pdb --verbose
```

## 向后兼容性

- 保持与原 `solvent_accessibility.py` 相同的命令行接口
- `solvent_accessibility_new.py` 提供兼容性入口
- 支持所有原有命令行参数

## 依赖

- Python 3.7+
- NumPy
- SciPy
- BioPython
- FreeSASA

## 许可证

MIT License