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

    # Bug C 修复（修正版）：导出 parameters + BN 的 weight/bias buffers，
    # 但排除 running_mean/running_var/num_batches_tracked（边端自适应统计不应覆盖）。
    # FrozenBatchNorm2d 将 weight/bias 注册为 buffer 而非 parameter，
    # 若不导出则注入后 Conv 权重与 BN 参数不匹配，模型崩溃。
    skip_suffixes = ('running_mean', 'running_var', 'num_batches_tracked')
    # 排除 adapter 参数：边端 adapter 经过在线适配有意义的状态，
    # 云端学生的 adapter 是随机初始化的，注入会覆盖边端适配进度
    skip_keywords = ('adapter',)
    fp16_state = {}
    for key, val in model.state_dict().items():
        if any(key.endswith(s) for s in skip_suffixes):
            continue
        if any(k in key for k in skip_keywords):
            continue
        fp16_state[key] = val.detach().half().cpu()

    # 先写临时文件再 rename，避免并发写入 crash
    tmp_path = save_path + '.tmp'
    if state_dict_only:
        torch.save(fp16_state, tmp_path)
    else:
        torch.save({'state_dict': fp16_state, 'model_class': type(model).__name__}, tmp_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    os.rename(tmp_path, save_path)

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

    state_dict = torch.load(buffer, map_location='cpu')

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

    def __init__(self, model: nn.Module, device: str = 'cpu', ema_alpha: float = 0.01):
        self.model = model
        self.device = device
        self.ema_alpha = ema_alpha  # EMA 混合系数：edge = (1-α)*edge + α*cloud
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

        # 只注入 box_predictor 参数，EMA 混合避免破坏边端适配状态
        current_state = self.model.state_dict()
        alpha = self.ema_alpha
        injected = 0
        for key, cloud_val in next_state.items():
            if 'box_predictor' not in key:
                continue
            if key not in current_state:
                continue
            edge_val = current_state[key]
            if cloud_val.shape == edge_val.shape and cloud_val.is_floating_point():
                current_state[key] = (1 - alpha) * edge_val + alpha * cloud_val.to(edge_val.device)
                injected += 1
        self.model.load_state_dict(current_state)
        logger.info(f"[DualBuffer] EMA 注入 {injected} 个 box_predictor 参数 (α={alpha})")

        self._swap_count += 1
        logger.info(f"[DualBuffer] 参数替换完成 (第 {self._swap_count} 次)")
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
