# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.v1.engine import infer_model_id


def test_explicit_override_takes_priority():
    sampling = MagicMock(name="sampling")
    pooling = MagicMock(name="pooling")
    assert (
        infer_model_id(sampling, pooling, explicit_model_id="custom") == "custom"
    )


def test_sampling_params_default_to_decode():
    sampling = MagicMock(name="sampling")
    assert infer_model_id(sampling, None) == "decode"


def test_pooling_params_use_task_name():
    pooling = MagicMock(name="pooling")
    pooling.task = "embed"
    assert infer_model_id(None, pooling) == "embed"


def test_pooling_other_task_name():
    pooling = MagicMock(name="pooling")
    pooling.task = "classify"
    assert infer_model_id(None, pooling) == "classify"


def test_neither_params_set_raises():
    with pytest.raises(AssertionError):
        infer_model_id(None, None)
