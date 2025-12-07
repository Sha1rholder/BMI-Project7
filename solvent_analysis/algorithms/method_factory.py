"""
方法工厂
"""

from typing import Union

from ..core.data_models import MethodType, AnalysisConfig
from .centroid_method import CentroidMethod
from .peratom_method import PerAtomMethod


class MethodFactory:
    """方法工厂"""

    @staticmethod
    def create_method(
        method_type: Union[MethodType, str],
        config: AnalysisConfig = None,
    ) -> Union[CentroidMethod, PerAtomMethod]:
        """
        创建分析方法

        Args:
            method_type: 方法类型（枚举或字符串）
            config: 分析配置

        Returns:
            Union[CentroidMethod, PerAtomMethod]: 分析方法实例
        """
        # 处理字符串输入
        if isinstance(method_type, str):
            method_type = MethodType(method_type.lower())

        if method_type == MethodType.CENTROID:
            return CentroidMethod(config)
        elif method_type == MethodType.PERATOM:
            return PerAtomMethod(config)
        else:
            raise ValueError(f"未知的方法类型: {method_type}")

    @staticmethod
    def get_available_methods() -> list:
        """获取可用方法列表"""
        return [method.value for method in MethodType]