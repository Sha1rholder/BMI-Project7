"""
测试重构后的代码
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")

    try:
        from solvent_analysis import ResidueInfo, WaterInfo, AnalysisConfig, MethodType
        print("  ✓ 核心数据模型导入成功")

        from solvent_analysis.core.distance_calculator import ChunkedDistanceCalculator
        print("  ✓ 距离计算模块导入成功")

        from solvent_analysis.core.accessibility_evaluator import CentroidEvaluator
        print("  ✓ 可及性评估模块导入成功")

        from solvent_analysis.io.pdb_loader import PDBLoader
        print("  ✓ PDB加载模块导入成功")

        from solvent_analysis.io.csv_writer import CSVWriter
        print("  ✓ CSV写入模块导入成功")

        from solvent_analysis.algorithms.centroid_method import CentroidMethod
        from solvent_analysis.algorithms.peratom_method import PerAtomMethod
        print("  ✓ 算法模块导入成功")

        from solvent_analysis.algorithms.freesasa_wrapper import FreeSASAWrapper
        print("  ✓ FreeSASA包装器导入成功")

        from solvent_analysis.cli.main import parse_args
        print("  ✓ 命令行接口导入成功")

        print("\n所有模块导入成功！")
        return True

    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def test_data_models():
    """测试数据模型"""
    print("\n测试数据模型...")

    try:
        import numpy as np
        from solvent_analysis import ResidueInfo, WaterInfo, AnalysisConfig

        # 测试ResidueInfo
        residue = ResidueInfo(
            chain="A",
            resnum=1,
            resname="ALA",
            coord=np.array([1.0, 2.0, 3.0])
        )
        print(f"  ✓ ResidueInfo创建成功: {residue}")

        # 测试WaterInfo
        waters = WaterInfo(
            coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            names=["HOH", "HOH"]
        )
        print(f"  ✓ WaterInfo创建成功: {waters.count}个水分子")

        # 测试AnalysisConfig
        config = AnalysisConfig(
            threshold=3.5,
            radius=5.0,
            chunk_size=5000
        )
        config.validate()
        print(f"  ✓ AnalysisConfig创建和验证成功")

        return True

    except Exception as e:
        print(f"  ✗ 数据模型测试失败: {e}")
        return False

def test_pdb_loader():
    """测试PDB加载器"""
    print("\n测试PDB加载器...")

    try:
        from solvent_analysis.io.pdb_loader import PDBLoader

        # 检查测试PDB文件是否存在
        test_pdb = Path("./pdb/SUMO1_water.pdb")
        if not test_pdb.exists():
            print(f"  ⚠ 测试PDB文件不存在: {test_pdb}")
            print("    跳过PDB加载测试")
            return True

        loader = PDBLoader(quiet=True)
        residues, waters, structure = loader.load(str(test_pdb))

        print(f"  ✓ PDB加载成功")
        print(f"    残基数: {len(residues)}")
        print(f"    水分子数: {waters.count}")
        print(f"    结构对象: {'存在' if structure else '不存在'}")

        if residues:
            print(f"    示例残基: {residues[0]}")

        return True

    except Exception as e:
        print(f"  ✗ PDB加载测试失败: {e}")
        return False

def test_cli_parsing():
    """测试命令行解析"""
    print("\n测试命令行解析...")

    try:
        from solvent_analysis.cli.main import parse_args

        # 测试基本参数解析
        test_args = [
            "--wet-pdb", "test_wet.pdb",
            "--dry-pdb", "test_dry.pdb",
            "--method", "peratom",
            "--threshold", "3.5",
            "--verbose"
        ]

        args = parse_args(test_args)

        print(f"  ✓ 命令行解析成功")
        print(f"    wet-pdb: {args.wet_pdb}")
        print(f"    dry-pdb: {args.dry_pdb}")
        print(f"    method: {args.method}")
        print(f"    threshold: {args.threshold}")
        print(f"    verbose: {args.verbose}")

        return True

    except Exception as e:
        print(f"  ✗ 命令行解析测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("测试重构后的溶剂可及性分析代码")
    print("=" * 60)

    tests = [
        ("模块导入", test_imports),
        ("数据模型", test_data_models),
        ("PDB加载器", test_pdb_loader),
        ("命令行解析", test_cli_parsing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {test_name}测试异常: {e}")

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        print("\n下一步:")
        print("1. 运行完整测试: python solvent_accessibility_new.py --wet-pdb ./pdb/SUMO1_water.pdb --dry-pdb ./pdb/SUMO1.pdb --verbose")
        print("2. 查看输出目录: ./output/")
        print("3. 比较新旧版本结果")
    else:
        print("⚠ 部分测试失败，请检查错误信息")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)