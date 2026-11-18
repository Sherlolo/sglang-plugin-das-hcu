import os
from pathlib import Path
import subprocess
import sys
import unittest

class DeepEPLowLatency(unittest.TestCase):
    def test_cpu_contracts_in_isolated_process(self):
        environment=os.environ.copy()
        environment.update({'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','HCU_ADAPTER_HOST':'','SGLANG_PLUGINS':'','PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        result=subprocess.run([sys.executable,'-B','-S',str(Path(__file__).with_name('kimi31_deepep_cpu_probe.py'))],env=environment,text=True,capture_output=True,timeout=120)
        if result.returncode==77:self.skipTest('CPU Torch is required')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
