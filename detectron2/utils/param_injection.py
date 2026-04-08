"""
边端参数注入模块
Edge-side Parameter Injection for Cloud-Edge Collaborative TTA.

实现论文 Chapter 5 的参数下发机制：
- FP16 序列化导出（~50MB）
- 双缓冲机制（保证边端推理不中断）
- 内存映射（mmap）快速加载
- MD5 校验完整性

关键指标（论文实测）：
- 参数导出: FP16 约 50MB
- 加载时间: ~8ms (mmap), ~12ms (总注入时间)
- 边端 FPS: 30+ (注入期间不停机)
"""

import hashlib
import io
import logging
import mmap
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# FP16 参数导出
# ---------------------------------------------------------------------------- #

def export_student_fp16(
    model: nn.Module,
    save_path: str,
    state_dict_only: bool = True,
) -> str:
    """
    将学生模型以 FP16 精度序列化保存。

    Args:
        model: 学生模型（全精度或FP16）
        save_path: 保存路径（.pt 文件）
        state_dict_only: 若 True 只保存 state_dict（推荐）
    Returns:
        save_path 和 MD5 校验码
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # 转为 FP16 state_dict
    fp16_state = {}
    for key, param in model.state_dict().items():
        fp16_state[key] = param.half()

    if state_dict_only:
        torch.save(fp16_state, save_path)
    else:
        torch.save({'state_dict': fp16_state, 'model_class': type(model).__name__}, save_path)

    # 计算MD5
    md5 = _compute_md5(save_path)
    md5_path = save_path + '.md5'
    with open(md5_path, 'w') as f:
        f.write(md5)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    logger.info(f"[ParamInjection] 导出 FP16 参数: {save_path} ({size_mb:.1f} MB), MD5: {md5}")
    return save_path


def _compute_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- #
# 内存映射加载
# ---------------------------------------------------------------------------- #

def mmap_load_state_dict(
    path: str,
    verify_md5: bool = True,
    target_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    使用内存映射快速加载模型参数。

    Args:
        path: FP16 模型参数文件路径
        verify_md5: 是否验证MD5
        target_dtype: 加载后转换的精度（推理时用 float32）
    Returns:
        state_dict
    """
    # MD5 验证
    if verify_md5:
        md5_path = path + '.md5'
        if os.path.exists(md5_path):
            with open(md5_path, 'r') as f:
                expected_md5 = f.read().strip()
            actual_md5 = _compute_md5(path)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"[ParamInjection] MD5 校验失败: 期望 {expected_md5}, 实际 {actual_md5}"
                )

    # 使用 mmap 加载
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buffer = io.BytesIO(mm.read())
        mm.close()

    state_dict = torch.load(buffer, map_location='cpu', weights_only=True)

    # 精度转换
    if target_dtype != torch.float16:
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

    return state_dict


# ---------------------------------------------------------------------------- #
# 双缓冲注入器
# ---------------------------------------------------------------------------- #

class DualBufferInjector:
    """
    双缓冲参数注入器。

    维护两个模型参数缓冲区（current 和 next），
    后台线程加载新参数到 next 缓冲，
    推理完成一帧后原子切换到 next。
    保证边端推理 30+ FPS 不中断。

    用法:
        injector = DualBufferInjector(edge_model)
        injector.prepare_next(new_param_path)   # 后台异步加载
        # 在每帧推理后调用:
        injector.try_swap()                     # 若新参数已就绪则切换
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self._lock = threading.Lock()
        self._next_state: Optional[Dict[str, torch.Tensor]] = None
        self._next_ready = False
        self._loading_thread: Optional[threading.Thread] = None
        self._swap_count = 0

    def prepare_next(
        self,
        param_path: str,
        target_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        后台异步加载新参数（不阻塞主线程推理）。
        """
        if self._loading_thread is not None and self._loading_thread.is_alive():
            logger.warning("[DualBuffer] 上一次加载尚未完成，跳过本次请求")
            return

        def _load_worker():
            try:
                state_dict = mmap_load_state_dict(param_path, target_dtype=target_dtype)
                with self._lock:
                    self._next_state = state_dict
                    self._next_ready = True
                logger.info(f"[DualBuffer] 新参数加载完毕: {param_path}")
            except Exception as e:
                logger.error(f"[DualBuffer] 参数加载失败: {e}")

        self._loading_thread = threading.Thread(target=_load_worker, daemon=True)
        self._loading_thread.start()

    def try_swap(self) -> bool:
        """
        若新参数已就绪，原子切换模型参数。
        应在每帧推理结束后调用（在 GIL 持有期间完成赋值，天然原子）。

        Returns:
            True 表示发生了切换
        """
        with self._lock:
            if not self._next_ready or self._next_state is None:
                return False
            next_state = self._next_state
            self._next_state = None
            self._next_ready = False

        # 加载新参数（CPU 上无需 cuda 同步）
        missing, unexpected = self.model.load_state_dict(next_state, strict=False)
        if missing:
            logger.debug(f"[DualBuffer] 缺少参数键: {missing[:5]}")
        if unexpected:
            logger.debug(f"[DualBuffer] 多余参数键: {unexpected[:5]}")

        self._swap_count += 1
        logger.info(f"[DualBuffer] 参数切换完成 (第 {self._swap_count} 次)")
        return True

    def inject_sync(self, param_path: str, target_dtype: torch.dtype = torch.float32) -> None:
        """
        同步注入参数（阻塞，用于测试或首次初始化）。
        """
        state_dict = mmap_load_state_dict(param_path, target_dtype=target_dtype)
        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"[DualBuffer] 同步注入完成: {param_path}")

    @property
    def swap_count(self) -> int:
        return self._swap_count


# ---------------------------------------------------------------------------- #
# 便捷函数
# ---------------------------------------------------------------------------- #

def inject_cloud_params(
    edge_model: nn.Module,
    cloud_param_path: str,
    async_mode: bool = True,
    injector: Optional[DualBufferInjector] = None,
) -> Optional[DualBufferInjector]:
    """
    便捷的参数注入接口。

    Args:
        edge_model: 边端模型
        cloud_param_path: 云端下发的参数文件路径
        async_mode: True 使用双缓冲异步注入，False 同步阻塞注入
        injector: 已有的 DualBufferInjector（异步模式复用）
    Returns:
        DualBufferInjector（异步模式）或 None（同步模式）
    """
    if not async_mode:
        state_dict = mmap_load_state_dict(cloud_param_path)
        edge_model.load_state_dict(state_dict, strict=False)
        return None

    if injector is None:
        injector = DualBufferInjector(edge_model)
    injector.prepare_next(cloud_param_path)
    return injector
