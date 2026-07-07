# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.worker.dual_model_runner import DualModelRunner
from vllm.v1.worker.gpu_worker import Worker


class DualModelWorker(Worker):
    """Experimental worker that loads one decode and one embed model."""

    def init_device(self):
        super().init_device()
        # super().init_device() built the worker's WorkStream(s) and wired them
        # onto the runner IT constructed. We replace that runner with the
        # dual-model runner here, so the WorkStreams must be re-wired onto it --
        # otherwise DualModelRunner._decode_ctx() raises "set_work_streams() must
        # be called before execute_model()".
        self.model_runner = DualModelRunner(self.vllm_config, self.device)
        assert self._work_stream is not None
        if self._aux_work_stream is not None:
            self.model_runner.set_work_streams(
                self._work_stream, self._aux_work_stream
            )
        else:
            self.model_runner.bind_main_stream(self._work_stream.stream)
