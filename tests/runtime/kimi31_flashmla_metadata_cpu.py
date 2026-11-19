from types import SimpleNamespace
import unittest
import torch

from sglang.srt.plugins import load_plugins

load_plugins()
from sglang.srt.layers.attention.dsa_backend import DSAFlashMLAMetadata
from hcu_sglang.layers.attention.current_dsa.flashmla_backend import DSAFlashMLAMetadata as HcuMetadata


class KimiMetadataTests(unittest.TestCase):
    def test_day0_registration_and_none_splits(self):
        self.assertIs(DSAFlashMLAMetadata, HcuMetadata)
        schedule = SimpleNamespace(have_initialized=True, num_splits=torch.tensor([1]))
        owner = DSAFlashMLAMetadata(schedule, None)
        view = owner.slice(slice(0, 9))
        self.assertIs(view.flashmla_metadata, schedule)
        self.assertIsNone(view.num_splits)
        view.copy_(DSAFlashMLAMetadata(SimpleNamespace(have_initialized=False), None))
        self.assertIs(owner.flashmla_metadata, schedule)
        self.assertTrue(owner.flashmla_metadata.have_initialized)
        self.assertIsNone(owner.num_splits)

    def test_tensor_copy_keeps_graph_addresses_and_slices(self):
        metadata = torch.zeros(2)
        splits = torch.zeros(4, dtype=torch.int32)
        owner = DSAFlashMLAMetadata(metadata, splits)
        view = owner.slice(slice(0, 2))
        metadata_ptr, splits_ptr = metadata.data_ptr(), splits.data_ptr()
        view.copy_(DSAFlashMLAMetadata(torch.ones(2), torch.ones(2, dtype=torch.int32)))
        self.assertEqual(owner.flashmla_metadata.data_ptr(), metadata_ptr)
        self.assertEqual(owner.num_splits.data_ptr(), splits_ptr)
        torch.testing.assert_close(metadata, torch.ones(2))
        torch.testing.assert_close(splits, torch.tensor([1, 1, 0, 0], dtype=torch.int32))


if __name__ == '__main__':
    unittest.main()
