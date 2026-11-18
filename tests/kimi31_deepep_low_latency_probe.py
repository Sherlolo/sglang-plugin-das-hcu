import os

if os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') != '1':
    raise SystemExit('Explicit GPU probe authorization is required')
if int(os.environ.get('WORLD_SIZE', '0')) != 16:
    raise SystemExit('This probe requires exactly 16 GPU ranks')

import datetime
import json
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from deep_ep import Buffer


def event(stage, **values):
    print(json.dumps({'rank': int(os.environ['RANK']), 'stage': stage, **values}), flush=True)


def main():
    rank=int(os.environ['RANK'])
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=90))
    buffer=None
    try:
        event('process_group_ready',world=dist.get_world_size())
        hidden=7168
        experts=896
        maximum=8
        integrated=os.environ.get('HCU_LL_INTEGRATED_PROBE')=='1'
        rdma=Buffer.get_low_latency_rdma_size_hint(maximum,hidden,16,experts,num_topk=16)
        nvl=max(config.get_nvl_buffer_size_hint(hidden*2,16) for config in [Buffer.get_dispatch_config(16),Buffer.get_combine_config(16)]) if integrated else 0
        event('buffer_initializing',rdma_bytes=rdma,mode='low_latency')
        buffer=Buffer(dist.group.WORLD,nvl,rdma,low_latency_mode=True,
                      num_qps_per_rank=56,explicitly_destroy=True)
        x=torch.full((maximum,hidden),float(rank+1),device='cuda',dtype=torch.bfloat16)
        ids=(torch.arange(16,device='cuda',dtype=torch.int64)[None,:].expand(maximum,-1)*56+rank).contiguous()
        weights=torch.full((maximum,16),1./16,device='cuda',dtype=torch.float32)
        if integrated:
            from hcu_sglang.layers.moe import ct_w4a8 as ct
            from hcu_sglang.layers.moe import deepep_current as D
            from deepgemm import pack_w4a8_moe_hipc_weight
            import inspect
            assert D._DEEPEP_LL_DISPATCH_USES_QUANT_TYPE
            assert 'quant_type' in inspect.signature(Buffer.low_latency_dispatch).parameters
            ct.register_ops()
            intermediate=2048
            local_experts=experts//16
            layer=SimpleNamespace(
                moe_runner_config=SimpleNamespace(activation='situ',swiglu_limit=None,gemm1_alpha=4.,gemm1_clamp_limit=25.),
                w13_weight=pack_w4a8_moe_hipc_weight(torch.full((local_experts,2*intermediate,hidden//2),17,device='cuda',dtype=torch.int8)),
                w2_weight=pack_w4a8_moe_hipc_weight(torch.full((local_experts,hidden,intermediate//2),17,device='cuda',dtype=torch.int8)),
                w13_weight_scale=torch.full((local_experts,2*intermediate,1),1e-4,device='cuda',dtype=torch.float32),
                w2_weight_scale=torch.full((local_experts,hidden,1),1e-3,device='cuda',dtype=torch.float32),
                w4a8_padded_intermediate_size=intermediate)
            D._hcu_deepgemm_is_selected=lambda:False
            impl=SimpleNamespace(quant_config={'hcu_w4a8_int8_dispatch':True},use_fp8=True,use_nvfp4=False,
                _get_buffer=lambda:buffer,_get_npu_mxfp_quantization_kwargs=lambda unused:{},
                num_max_dispatch_tokens_per_rank=maximum,num_experts=experts,return_recv_hook=True)
        def step():
            if integrated:
                received,counts,ready,hook=D._DeepEPDispatcherImplLowLatency._dispatch_core(impl,x,ids,weights)
                handle=impl.handle
            else:
                received,counts,handle,ready,hook=buffer.low_latency_dispatch(x,ids,weights,maximum,experts,quant_type=1,return_recv_hook=True)
            hook()
            q,scales=received
            assert q.dtype==torch.int8 and scales.dtype==torch.float32
            if not torch.cuda.is_current_stream_capturing():
                event('wire_contract',activation_dtype=str(q.dtype),scale_dtype=str(scales.dtype),activation_shape=list(q.shape),scale_shape=list(scales.shape))
            if integrated:
                dispatch=SimpleNamespace(hidden_states=q,hidden_states_scale=scales,masked_m=counts,expected_m=3)
                partial=ct.forward_masked(layer,dispatch)
            else:
                partial=(q.float()*scales).bfloat16()
            output,ready,hook=buffer.low_latency_combine(partial,ids,weights,handle,return_recv_hook=True)
            hook()
            return output
        for _ in range(3):output=step()
        torch.cuda.synchronize()
        if integrated:
            assert torch.isfinite(output).all() and output.abs().max()>0
        else:
            torch.testing.assert_close(output,x,rtol=.01,atol=.01)
        event('eager_passed')
        def normal_round_trip():
            layout=buffer.get_dispatch_layout(ids,experts)
            received=buffer.dispatch(x,num_tokens_per_rank=layout[0],num_tokens_per_rdma_rank=layout[1],num_tokens_per_expert=layout[2],is_token_in_rank=layout[3],topk_idx=ids,topk_weights=weights,expert_alignment=256)
            recv_x,_,_,_,handle,_=received
            normal,_,_=buffer.combine(recv_x,handle)
            torch.testing.assert_close(normal,x*16,rtol=0,atol=0)
            from unittest.mock import patch
            state=SimpleNamespace(buffer=buffer,num_max_dispatch_tokens_per_rank=maximum,hidden_size=hidden,num_experts=experts,num_topk=16)
            with patch.object(D.DeepEPBuffer,'_state',classmethod(lambda cls:state)):
                D.DeepEPBuffer.clean_buffer()
        if integrated:
            normal_round_trip()
            event('normal_to_low_latency_clean_passed',num_topk=16)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):output=step()
        for offset in [0,1,2]:
            x.fill_(rank+1+offset)
            reference=step().clone() if integrated else x
            if integrated:normal_round_trip()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,reference,rtol=.01,atol=.01)
        event('graph_passed')
        dist.barrier()
        event('passed',scope='adapter + plugin INT8 dispatch + masked HIPC SiTU + native combine graph replay' if integrated else 'native DeepEP INT8 low-latency eager and graph replay; no model or HIPC math')
    finally:
        if buffer is not None:buffer.destroy()
        dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException:
        event('failed',traceback=traceback.format_exc())
        raise
