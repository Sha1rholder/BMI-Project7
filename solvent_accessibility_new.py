"""
向后兼容的溶剂可及性分析脚本
保持与原solvent_accessibility.py相同的命令行接口
"""

import sys
from solvent_analysis.cli.main import main

if __name__ == "__main__":
    sys.exit(main())