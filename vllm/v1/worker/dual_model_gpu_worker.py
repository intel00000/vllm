# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker


class DualModelWorker(Worker):
    """Compatibility alias for --worker-cls callers.

    Worker.init_device selects DualModelRunner via _select_model_runner_class
    and wires the WorkStreams onto it; the old init_device override here built
    a second runner instance and leaked the first one's executor threads.
    """
