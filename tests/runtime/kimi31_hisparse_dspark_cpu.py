from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from hcu_sglang.kimi31_hisparse_dspark import commit_verify, prepare_verify


def test_verify_slots_are_prepared_then_committed_by_chain_offset():
    coordinator = Mock()
    batch = SimpleNamespace(hisparse_coordinator=coordinator)
    worker = SimpleNamespace(verify_num_draft_tokens=8)
    verify_input = SimpleNamespace(ragged_verify_layout=None)

    prepare_verify(worker, batch=batch, verify_input=verify_input)
    commit_verify(
        worker, batch=batch, seq_lens_pre_verify=None,
        seq_lens_post_verify=None, commit_lens=torch.tensor([1, 3, 8]),
    )

    assert [call[0] for call in coordinator.mock_calls] == [
        "prepare_spec_verify", "commit_spec_accept_tokens",
    ]
    coordinator.prepare_spec_verify.assert_called_once_with(batch)
    assert coordinator.commit_spec_accept_tokens.call_args.kwargs["batch"] is batch
    indices = coordinator.commit_spec_accept_tokens.call_args.kwargs["accept_indices"]
    assert indices.tolist() == [
        [0, -1, -1, -1, -1, -1, -1, -1],
        [8, 9, 10, -1, -1, -1, -1, -1],
        [16, 17, 18, 19, 20, 21, 22, 23],
    ]


def test_rejects_ragged_verify_before_mapping():
    coordinator = Mock()
    with pytest.raises(RuntimeError, match="static verify layout"):
        prepare_verify(
            None,
            batch=SimpleNamespace(hisparse_coordinator=coordinator),
            verify_input=SimpleNamespace(ragged_verify_layout=object()),
        )
    assert not coordinator.mock_calls
