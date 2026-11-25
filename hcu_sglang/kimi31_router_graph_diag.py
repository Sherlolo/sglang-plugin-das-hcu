"""Read kernel symbols and launch dimensions from a live DTK HIP graph."""
import ctypes as C


class Dim3(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("z", C.c_uint)]


class KernelParams(C.Structure):
    _fields_ = [
        ("blockDim", Dim3), ("extra", C.c_void_p), ("func", C.c_void_p),
        ("gridDim", Dim3), ("kernelParams", C.c_void_p), ("sharedMemBytes", C.c_uint),
    ]


def kernels(graph_handle):
    hip = C.CDLL("/opt/dtk/lib/libamdhip64.so")

    def bind(name, *args):
        function = getattr(hip, name)
        function.argtypes = args
        function.restype = C.c_int
        return function

    get_nodes = bind("hipGraphGetNodes", C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t))
    get_type = bind("hipGraphNodeGetType", C.c_void_p, C.POINTER(C.c_int))
    get_params = bind("hipGraphKernelNodeGetParams", C.c_void_p, C.POINTER(KernelParams))
    get_name = bind("hipFuncGetName", C.POINTER(C.c_char_p), C.c_void_p)
    get_child = bind("hipGraphChildGraphNodeGetGraph", C.c_void_p, C.POINTER(C.c_void_p))

    def check(status, name):
        if status != 0:
            raise RuntimeError(f"{name} returned HIP error {status}")

    def visit(handle):
        count = C.c_size_t()
        check(get_nodes(handle, None, C.byref(count)), "hipGraphGetNodes(count)")
        nodes = (C.c_void_p * count.value)()
        check(get_nodes(handle, nodes, C.byref(count)), "hipGraphGetNodes(nodes)")
        for node in nodes[:count.value]:
            node_type = C.c_int()
            check(get_type(node, C.byref(node_type)), "hipGraphNodeGetType")
            if node_type.value == 4:
                subgraph = C.c_void_p()
                check(get_child(node, C.byref(subgraph)), "hipGraphChildGraphNodeGetGraph")
                yield from visit(subgraph)
            elif node_type.value == 0:
                params = KernelParams()
                check(get_params(node, C.byref(params)), "hipGraphKernelNodeGetParams")
                name = C.c_char_p()
                status = get_name(C.byref(name), params.func)
                yield {
                    "node_type": node_type.value,
                    "name": name.value.decode() if status == 0 and name.value else None,
                    "name_status": status,
                    "grid": [params.gridDim.x, params.gridDim.y, params.gridDim.z],
                    "block": [params.blockDim.x, params.blockDim.y, params.blockDim.z],
                }

            else:
                yield {"node_type": node_type.value}

    return list(visit(graph_handle))


def capture_one(original, backend, shape_key, *args, **kwargs):
    """Retain and inspect the target Verify graph on TP0 for diagnostics."""
    from functools import partial
    import json
    import os
    from pathlib import Path

    import torch

    runner = backend._cuda_graph_runner
    mode = getattr(runner, "capture_forward_mode", None)
    if (
        shape_key.size != 8
        or backend._tp_group.rank_in_group != 0
        or mode is None
        or not mode.is_target_verify()
        or runner.model_runner.is_draft_worker
    ):
        return original(backend, shape_key, *args, **kwargs)

    original_graph_type = torch.cuda.CUDAGraph

    torch.cuda.CUDAGraph = partial(original_graph_type, keep_graph=True)
    try:
        result = original(backend, shape_key, *args, **kwargs)
    finally:
        torch.cuda.CUDAGraph = original_graph_type
    graph = backend._graphs[shape_key]
    graph.instantiate()
    nodes = kernels(graph.raw_cuda_graph())
    directory = Path(os.environ["HCU_KIMI31_ROUTER_GRAPH_DUMP_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    graph_id = len(backend._graphs)
    trace_id = f"target-verify-bs8-{os.getpid()}-{graph_id}"
    report = directory / f"{trace_id}.json"
    report.write_text(json.dumps({"graph": trace_id, "shape": repr(shape_key), "nodes": nodes}))
    graph._hcu_kimi31_router_trace_id = trace_id
    return result


def replay(original, backend, shape_key, *args, **kwargs):
    graph = backend._graphs.get(shape_key)
    trace_id = getattr(graph, "_hcu_kimi31_router_trace_id", None)
    if not trace_id:
        return original(backend, shape_key, *args, **kwargs)

    import torch

    with torch.profiler.record_function(f"router-graph:{trace_id}"):
        return original(backend, shape_key, *args, **kwargs)
