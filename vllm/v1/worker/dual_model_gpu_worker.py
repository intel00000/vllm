# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.worker.dual_model_runner import DualModelRunner
from vllm.v1.worker.gpu_worker import Worker


class DualModelWorker(Worker):
    """Experimental worker that loads one decode and one embed model."""

    def init_device(self):
        super().init_device()
        self.model_runner = DualModelRunner(self.vllm_config, self.device)
