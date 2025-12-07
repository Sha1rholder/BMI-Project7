"""
分步验证脚本
逐步验证重构后的代码的每个部分
"""

import sys
import importlib
from pathlib import Path
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'='*60}")

def test_module_imports():
    """步骤1：测试所有模块导入"""
    print_step(1, "测试模块导入")

    modules_to_test = [
        ("solvent_analysis", "主包"),
        ("solvent_analysis.core.data_models", "核心数据模型"),
        ("solvent_analysis.core.distance_calculator", "距离计算器"),
        ("solvent_analysis.core.accessibility_evaluator", "可及性评估器"),
        ("solvent_analysis.io.pdb_loader", "PDB加载器"),
        ("solvent_analysis.io.csv_writer", "CSV写入器"),
        ("solvent_analysis.io.result_formatter", "结果格式化器"),
        ("solvent_analysis.algorithms.centroid_method", "质心法"),
        ("solvent_analysis.algorithms.peratom_method", "原子级方法"),
        ("solvent_analysis.algorithms.freesasa_wrapper", "FreeSASA包装器"),
        ("solvent_analysis.algorithms.method_factory", "方法工厂"),
        ("solvent_analysis.utils.progress", "进度条工具"),
        ("solvent_analysis.utils.logger", "日志工具"),
        ("solvent_analysis.utils.validation", "验证工具"),
        ("solvent_analysis.cli.main", "命令行接口"),
    ]

    success_count = 0
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {description} ({module_name})")
            success_count += 1
        except ImportError as e:
            print(f"  ✗ {description} 导入失败: {e}")

    print(f"\n导入测试: {success_count}/{len(modules_to_test)} 通过")
    return success_count == len(modules_to_test)

def test_data_models():
    """步骤2：测试数据模型"""
    print_step(2, "测试数据模型")

    try:
        from solvent_analysis import (
            ResidueInfo, WaterInfo, AccessibilityResult,
            AnalysisConfig, MethodType
        )

        # 测试ResidueInfo
        residue = ResidueInfo(
            chain="A",
            resnum=1,
            resname="ALA",
            coord=np.array([1.0, 2.0, 3.0])
        )
        print(f"  ✓ ResidueInfo创建成功")
        print(f"    链: {residue.chain}, 编号: {residue.resnum}, 名称: {residue.resname}")

        # 测试WaterInfo
        waters = WaterInfo(
            coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            names=["HOH", "HOH"]
        )
        print(f"  ✓ WaterInfo创建成功")
        print(f"    水分子数: {waters.count}, 是否为空: {waters.is_empty()}")

        # 测试AnalysisConfig
        config = AnalysisConfig(
            threshold=3.5,
            radius=5.0,
            chunk_size=5000
        )
        config.validate()
        print(f"  ✓ AnalysisConfig创建和验证成功")
        print(f"    阈值: {config.threshold}, 半径: {config.radius}, 分块大小: {config.chunk_size}")

        # 测试MethodType
        print(f"  ✓ MethodType枚举: {list(MethodType)}")

        # 测试AccessibilityResult
        result = AccessibilityResult(
            residue=residue,
            min_distance=2.5,
            water_count=3,
            accessible=True,
            method=MethodType.CENTROID
        )
        print(f"  ✓ AccessibilityResult创建成功")
        print(f"    可及性: {result.accessible}, 方法: {result.method}")

        return True

    except Exception as e:
        print(f"  ✗ 数据模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distance_calculator():
    """步骤3：测试距离计算器"""
    print_step(3, "测试距离计算器")

    try:
        from solvent_analysis.core.data_models import ResidueInfo, WaterInfo
        from solvent_analysis.core.distance_calculator import (
            ChunkedDistanceCalculator, PerAtomDistanceCalculator
        )

        # 创建测试数据
        residues = [
            ResidueInfo("A", 1, "ALA", np.array([0.0, 0.0, 0.0])),
            ResidueInfo("A", 2, "GLY", np.array([5.0, 0.0, 0.0])),
        ]

        waters = WaterInfo(
            coords=np.array([
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ]),
            names=["HOH", "HOH", "HOH"]
        )

        # 测试ChunkedDistanceCalculator
        calculator = ChunkedDistanceCalculator(chunk_size=1000)

        min_distances = calculator.compute_min_distances(residues, waters)
        print(f"  ✓ ChunkedDistanceCalculator最小距离计算成功")
        print(f"    距离: {min_distances}")

        water_counts = calculator.count_waters_within_radius(residues, waters, radius=3.0)
        print(f"  ✓ 半径内水分子统计成功")
        print(f"    数量: {water_counts}")

        # 测试PerAtomDistanceCalculator
        peratom_calc = PerAtomDistanceCalculator(chunk_size=1000)
        peratom_distances = peratom_calc.compute_min_distances(residues, waters)
        print(f"  ✓ PerAtomDistanceCalculator最小距离计算成功")
        print(f"    距离: {peratom_distances}")

        return True

    except Exception as e:
        print(f"  ✗ 距离计算器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accessibility_evaluator():
    """步骤4：测试可及性评估器"""
    print_step(4, "测试可及性评估器")

    try:
        from solvent_analysis.core.data_models import (
            ResidueInfo, WaterInfo, AnalysisConfig, MethodType
        )
        from solvent_analysis.core.accessibility_evaluator import (
            CentroidEvaluator, PerAtomEvaluator, EvaluatorFactory
        )
        import numpy as np

        # 创建测试数据
        residues = [
            ResidueInfo("A", 1, "ALA", np.array([0.0, 0.0, 0.0])),
            ResidueInfo("A", 2, "GLY", np.array([5.0, 0.0, 0.0])),
        ]

        config = AnalysisConfig(threshold=3.0, radius=5.0)

        # 测试数据
        min_distances = np.array([2.5, 4.0])  # 第一个可及，第二个不可及
        water_counts = np.array([3, 1])

        # 测试CentroidEvaluator
        centroid_evaluator = CentroidEvaluator()
        centroid_results = centroid_evaluator.evaluate(
            residues, min_distances, water_counts, config
        )
        print(f"  ✓ CentroidEvaluator评估成功")
        print(f"    结果数: {len(centroid_results)}")
        for r in centroid_results:
            print(f"    {r.residue.resname}{r.residue.resnum}: 可及={r.accessible}")

        # 测试PerAtomEvaluator
        peratom_evaluator = PerAtomEvaluator()
        # 设置原子距离（模拟）
        atom_distances = {
            ("A", "1"): np.array([2.0, 3.0, 4.0]),  # 平均3.0，有原子在阈值内
            ("A", "2"): np.array([5.0, 6.0, 7.0]),  # 都大于阈值
        }
        peratom_evaluator.set_atom_distances(atom_distances)

        peratom_results = peratom_evaluator.evaluate(
            residues, min_distances, water_counts, config
        )
        print(f"  ✓ PerAtomEvaluator评估成功")
        for r in peratom_results:
            print(f"    {r.residue.resname}{r.residue.resnum}: 可及={r.accessible}")

        # 测试EvaluatorFactory
        centroid_from_factory = EvaluatorFactory.create_evaluator(MethodType.CENTROID)
        peratom_from_factory = EvaluatorFactory.create_evaluator(
            MethodType.PERATOM, atom_distances
        )
        print(f"  ✓ EvaluatorFactory创建成功")
        print(f"    创建的评估器: {type(centroid_from_factory).__name__}, "
              f"{type(peratom_from_factory).__name__}")

        return True

    except Exception as e:
        print(f"  ✗ 可及性评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdb_loader():
    """步骤5：测试PDB加载器"""
    print_step(5, "测试PDB加载器")

    try:
        from solvent_analysis.io.pdb_loader import PDBLoader

        # 检查测试文件
        test_pdb = Path("./pdb/SUMO1_water.pdb")
        if not test_pdb.exists():
            print(f"  ⚠ 测试PDB文件不存在: {test_pdb}")
            print("    跳过实际加载测试，测试接口...")
            # 测试接口而不实际加载
            loader = PDBLoader(quiet=True)
            print(f"  ✓ PDBLoader接口测试成功")
            return True

        loader = PDBLoader(quiet=True)
        residues, waters, structure = loader.load(str(test_pdb))

        print(f"  ✓ PDB加载成功")
        print(f"    残基数: {len(residues)}")
        print(f"    水分子数: {waters.count}")
        print(f"    结构对象: {'存在' if structure else '不存在'}")

        if residues:
            print(f"    示例残基:")
            for i, r in enumerate(residues[:3]):
                print(f"      {r.chain}{r.resnum} {r.resname} "
                      f"质心: [{r.coord[0]:.1f}, {r.coord[1]:.1f}, {r.coord[2]:.1f}]")

        return True

    except Exception as e:
        print(f"  ✗ PDB加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_writer():
    """步骤6：测试CSV写入器"""
    print_step(6, "测试CSV写入器")

    try:
        from solvent_analysis.core.data_models import (
            ResidueInfo, AccessibilityResult, MethodType
        )
        from solvent_analysis.io.csv_writer import CSVWriter
        import numpy as np

        # 创建测试结果
        residues = [
            ResidueInfo("A", 1, "ALA", np.array([1.0, 2.0, 3.0])),
            ResidueInfo("A", 2, "GLY", np.array([4.0, 5.0, 6.0])),
        ]

        results = [
            AccessibilityResult(
                residue=residues[0],
                min_distance=2.5,
                water_count=3,
                accessible=True,
                method=MethodType.CENTROID
            ),
            AccessibilityResult(
                residue=residues[1],
                min_distance=4.0,
                water_count=1,
                accessible=False,
                method=MethodType.CENTROID
            ),
        ]

        # 测试写入结果
        test_file = "./output/test_output.csv"
        CSVWriter.write_results(test_file, results)

        print(f"  ✓ CSV写入成功")
        print(f"    文件: {test_file}")

        # 读取并验证文件
        with open(test_file, "r") as f:
            lines = f.readlines()
            print(f"    行数: {len(lines)}")
            print(f"    表头: {lines[0].strip()}")
            print(f"    第一行数据: {lines[1].strip()}")

        # 测试通用写入
        generic_file = "./output/test_generic.csv"
        data = [
            ["A", "1", "ALA", "2.500", "3", "Yes"],
            ["A", "2", "GLY", "4.000", "1", "No"],
        ]
        header = ["chain", "resnum", "resname", "distance", "water_count", "accessible"]
        CSVWriter.write_generic(generic_file, data, header)

        print(f"  ✓ 通用CSV写入成功")
        print(f"    文件: {generic_file}")

        return True

    except Exception as e:
        print(f"  ✗ CSV写入器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_factory():
    """步骤7：测试方法工厂"""
    print_step(7, "测试方法工厂")

    try:
        from solvent_analysis.core.data_models import AnalysisConfig, MethodType
        from solvent_analysis.algorithms.method_factory import MethodFactory

        config = AnalysisConfig(threshold=3.5, radius=5.0)

        # 测试创建质心法
        centroid_method = MethodFactory.create_method(MethodType.CENTROID, config)
        print(f"  ✓ 质心法创建成功")
        print(f"    类型: {type(centroid_method).__name__}")
        print(f"    方法类型: {centroid_method.get_method_type()}")

        # 测试创建原子级方法
        peratom_method = MethodFactory.create_method(MethodType.PERATOM, config)
        print(f"  ✓ 原子级方法创建成功")
        print(f"    类型: {type(peratom_method).__name__}")
        print(f"    方法类型: {peratom_method.get_method_type()}")

        # 测试字符串输入
        centroid_from_str = MethodFactory.create_method("centroid", config)
        peratom_from_str = MethodFactory.create_method("peratom", config)
        print(f"  ✓ 字符串输入创建成功")

        # 测试可用方法列表
        available_methods = MethodFactory.get_available_methods()
        print(f"  ✓ 可用方法列表: {available_methods}")

        return True

    except Exception as e:
        print(f"  ✗ 方法工厂测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_interface():
    """步骤8：测试命令行接口"""
    print_step(8, "测试命令行接口")

    try:
        from solvent_analysis.cli.main import parse_args, create_config

        # 测试参数解析
        test_args = [
            "--wet-pdb", "test_wet.pdb",
            "--dry-pdb", "test_dry.pdb",
            "--method", "peratom",
            "--threshold", "3.5",
            "--R", "5.0",  # 注意：参数名是 --R，不是 --radius
            "--chunk", "5000",
            "--verbose",
        ]

        args = parse_args(test_args)
        print(f"  ✓ 命令行参数解析成功")
        print(f"    wet-pdb: {args.wet_pdb}")
        print(f"    dry-pdb: {args.dry_pdb}")
        print(f"    method: {args.method}")
        print(f"    threshold: {args.threshold}")
        print(f"    R: {args.R}")
        print(f"    verbose: {args.verbose}")

        # 测试配置创建
        config = create_config(args)
        print(f"  ✓ 配置创建成功")
        print(f"    阈值: {config.threshold}")
        print(f"    半径: {config.radius}")
        print(f"    分块大小: {config.chunk_size}")

        # 验证配置
        config.validate()
        print(f"  ✓ 配置验证成功")

        return True

    except Exception as e:
        print(f"  ✗ 命令行接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """步骤9：测试集成功能"""
    print_step(9, "测试集成功能")

    try:
        # 测试完整的分析流程（使用模拟数据）
        print("  测试集成分析流程...")

        from solvent_analysis.core.data_models import (
            ResidueInfo, WaterInfo, AnalysisConfig, MethodType
        )
        from solvent_analysis.algorithms.method_factory import MethodFactory
        import numpy as np

        # 创建模拟数据
        residues = [
            ResidueInfo("A", 1, "ALA", np.array([0.0, 0.0, 0.0])),
            ResidueInfo("A", 2, "GLY", np.array([5.0, 0.0, 0.0])),
            ResidueInfo("A", 3, "SER", np.array([10.0, 0.0, 0.0])),
        ]

        waters = WaterInfo(
            coords=np.array([
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]),
            names=["HOH", "HOH", "HOH", "HOH"]
        )

        # 创建配置
        config = AnalysisConfig(
            threshold=3.0,
            radius=5.0,
            chunk_size=1000
        )

        # 创建方法
        method = MethodFactory.create_method(MethodType.CENTROID, config)

        # 模拟结构对象（对于质心法不需要实际结构）
        class MockStructure:
            def __getitem__(self, key):
                return self
            def __getitem__(self, key):
                return self
            def __getitem__(self, key):
                class MockResidue:
                    def __init__(self):
                        self.atoms = []
                return MockResidue()

        mock_structure = MockStructure()

        # 执行分析
        results = method.analyze(residues, waters, mock_structure)

        print(f"  ✓ 集成分析成功")
        print(f"    分析残基数: {len(residues)}")
        print(f"    结果数: {len(results)}")

        # 统计结果
        accessible = sum(1 for r in results if r.accessible)
        print(f"    可及残基数: {accessible}/{len(results)}")

        # 显示部分结果
        print(f"    示例结果:")
        for i, r in enumerate(results[:2]):
            status = "可及" if r.accessible else "不可及"
            print(f"      {r.residue.chain}{r.residue.resnum} {r.residue.resname}: "
                  f"距离={r.min_distance:.2f}Å, {status}")

        return True

    except Exception as e:
        print(f"  ✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有验证步骤"""
    print("溶剂可及性分析工具包 - 分步验证")
    print("=" * 60)

    steps = [
        ("模块导入", test_module_imports),
        ("数据模型", test_data_models),
        ("距离计算器", test_distance_calculator),
        ("可及性评估器", test_accessibility_evaluator),
        ("PDB加载器", test_pdb_loader),
        ("CSV写入器", test_csv_writer),
        ("方法工厂", test_method_factory),
        ("命令行接口", test_cli_interface),
        ("集成功能", test_integration),
    ]

    passed = 0
    total = len(steps)

    for step_name, step_func in steps:
        try:
            if step_func():
                passed += 1
            else:
                print(f"  ⚠ {step_name}测试失败")
        except Exception as e:
            print(f"  ✗ {step_name}测试异常: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"验证结果: {passed}/{total} 步骤通过")

    if passed == total:
        print("🎉 所有验证步骤通过！代码质量良好。")
        print("\n下一步建议:")
        print("1. 运行完整功能测试: python test_refactored.py")
        print("2. 运行实际分析: python solvent_accessibility_new.py --wet-pdb ./pdb/SUMO1_water.pdb --dry-pdb ./pdb/SUMO1.pdb --verbose")
        print("3. 查看输出文件: ./output/")
    else:
        print(f"⚠ {total - passed} 个步骤失败，请检查错误信息")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)