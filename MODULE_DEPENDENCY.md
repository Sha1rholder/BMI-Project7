# 模块依赖关系图

## 整体依赖关系
```
命令行接口 (cli/main.py)
    │
    ├── 算法模块 (algorithms/)
    │   ├── 质心法 (centroid_method.py)
    │   ├── 原子级方法 (peratom_method.py)
    │   ├── FreeSASA包装器 (freesasa_wrapper.py)
    │   └── 方法工厂 (method_factory.py)
    │
    ├── 输入输出模块 (io/)
    │   ├── PDB加载器 (pdb_loader.py)
    │   ├── CSV写入器 (csv_writer.py)
    │   └── 结果格式化器 (result_formatter.py)
    │
    ├── 核心模块 (core/)
    │   ├── 数据模型 (data_models.py)
    │   ├── 距离计算器 (distance_calculator.py)
    │   └── 可及性评估器 (accessibility_evaluator.py)
    │
    └── 工具模块 (utils/)
        ├── 进度条 (progress.py)
        ├── 日志工具 (logger.py)
        └── 验证工具 (validation.py)
```

## 详细依赖关系

### 1. 核心模块 (core/)
```
data_models.py
    ├── 依赖: numpy, dataclasses, typing, enum
    ├── 导出: ResidueInfo, WaterInfo, AccessibilityResult, AnalysisConfig, MethodType
    └── 被依赖: 所有其他模块

distance_calculator.py
    ├── 依赖: data_models, scipy.spatial.KDTree, numpy, abc
    ├── 导出: DistanceCalculator, ChunkedDistanceCalculator, PerAtomDistanceCalculator
    └── 被依赖: algorithms/*, accessibility_evaluator.py

accessibility_evaluator.py
    ├── 依赖: data_models, distance_calculator, numpy, abc
    ├── 导出: AccessibilityEvaluator, CentroidEvaluator, PerAtomEvaluator, EvaluatorFactory
    └── 被依赖: algorithms/*
```

### 2. 算法模块 (algorithms/)
```
centroid_method.py
    ├── 依赖: core.data_models, core.distance_calculator, core.accessibility_evaluator
    ├── 导出: CentroidMethod
    └── 被依赖: method_factory.py, cli/main.py

peratom_method.py
    ├── 依赖: core.data_models, core.distance_calculator, core.accessibility_evaluator
    ├── 导出: PerAtomMethod
    └── 被依赖: method_factory.py, cli/main.py

freesasa_wrapper.py
    ├── 依赖: core.data_models, freesasa
    ├── 导出: FreeSASAWrapper
    └── 被依赖: cli/main.py

method_factory.py
    ├── 依赖: core.data_models, centroid_method, peratom_method
    ├── 导出: MethodFactory
    └── 被依赖: cli/main.py
```

### 3. 输入输出模块 (io/)
```
pdb_loader.py
    ├── 依赖: core.data_models, Bio.PDB, numpy
    ├── 导出: PDBLoader, load_pdb
    └── 被依赖: cli/main.py, algorithms/*

csv_writer.py
    ├── 依赖: core.data_models, csv, pathlib
    ├── 导出: CSVWriter
    └── 被依赖: cli/main.py, result_formatter.py

result_formatter.py
    ├── 依赖: core.data_models, csv_writer
    ├── 导出: ResultFormatter
    └── 被依赖: cli/main.py
```

### 4. 工具模块 (utils/)
```
progress.py
    ├── 依赖: sys, time
    ├── 导出: ProgressBar
    └── 被依赖: 可选，任何需要进度显示的地方

logger.py
    ├── 依赖: logging, sys, pathlib
    ├── 导出: setup_logger, get_logger, LogMixin
    └── 被依赖: 可选，任何需要日志的地方

validation.py
    ├── 依赖: core.data_models, pathlib
    ├── 导出: validate_pdb_file, validate_config, validate_output_dir
    └── 被依赖: cli/main.py
```

### 5. 命令行接口 (cli/)
```
main.py
    ├── 依赖: 所有其他模块
    ├── 导出: parse_args, main
    └── 被依赖: solvent_accessibility_new.py, __main__.py
```

## 数据流图

```
用户输入
    ↓
命令行参数解析 (cli/main.py:parse_args)
    ↓
配置创建 (cli/main.py:create_config)
    ↓
PDB加载 (io/pdb_loader.py:PDBLoader.load)
    ├── 残基列表 (List[ResidueInfo])
    ├── 水分子信息 (WaterInfo)
    └── 结构对象 (BioPython Structure)
    ↓
方法选择 (algorithms/method_factory.py:MethodFactory.create_method)
    ↓
距离计算 (core/distance_calculator.py:DistanceCalculator.compute_min_distances)
    ↓
水分子统计 (core/distance_calculator.py:DistanceCalculator.count_waters_within_radius)
    ↓
可及性评估 (core/accessibility_evaluator.py:AccessibilityEvaluator.evaluate)
    ↓
结果格式化 (io/result_formatter.py:ResultFormatter)
    ↓
文件输出 (io/csv_writer.py:CSVWriter.write_results)
    ↓
FreeSASA对比 (可选) (algorithms/freesasa_wrapper.py:FreeSASAWrapper.compute_residue_sasa)
    ↓
对比结果输出 (io/csv_writer.py:CSVWriter.write_comparison)
    ↓
用户输出
```

## 关键接口定义

### 1. 距离计算器接口
```python
class DistanceCalculator(ABC):
    def compute_min_distances(self, residues: List[ResidueInfo], waters: WaterInfo) -> np.ndarray
    def count_waters_within_radius(self, residues: List[ResidueInfo], waters: WaterInfo, radius: float) -> np.ndarray
```

### 2. 可及性评估器接口
```python
class AccessibilityEvaluator(ABC):
    def evaluate(self, residues: List[ResidueInfo],
                 min_distances: np.ndarray,
                 water_counts: np.ndarray,
                 config: AnalysisConfig) -> List[AccessibilityResult]
```

### 3. 分析方法接口（隐式）
```python
# 所有分析方法都应实现
def analyze(self, residues: List[ResidueInfo],
            waters: WaterInfo,
            structure) -> List[AccessibilityResult]
def get_method_type(self) -> MethodType
```

## 验证依赖关系的方法

### 1. 检查导入语句
```python
# 在每个文件中查看import语句
# 确保没有循环依赖
```

### 2. 运行依赖检查
```bash
# 使用pydeps生成依赖图
pip install pydeps
pydeps solvent_analysis --show-dot | dot -Tpng > dependencies.png
```

### 3. 测试模块独立性
```python
# 尝试单独导入每个模块
# 确保没有缺失的依赖
```

## 设计原则验证

1. **单一职责原则**：每个模块/类只负责一个功能
2. **开闭原则**：通过接口扩展，而不是修改现有代码
3. **依赖倒置原则**：依赖抽象，而不是具体实现
4. **接口隔离原则**：小而专一的接口
5. **里氏替换原则**：子类可以替换父类

这个依赖关系图帮助你理解代码的组织结构和模块间的交互方式。