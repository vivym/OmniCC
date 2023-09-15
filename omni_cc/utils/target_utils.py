import logging

import tvm
from tvm.target import Target

logger = logging.getLogger(__file__)


def detect_local_metal_host():
    target_triple: str = tvm._ffi.get_global_func("tvm.codegen.llvm.GetDefaultTargetTriple")()
    process_triple: str = tvm._ffi.get_global_func("tvm.codegen.llvm.GetProcessTriple")()
    host_cpu: int = tvm._ffi.get_global_func("tvm.codegen.llvm.GetHostCPUName")()
    logger.info(
        f"Host CPU dection:\n  Target triple: {target_triple}\n  Process triple: {process_triple}\n  Host CPU: {host_cpu}"
    )
    if target_triple.startswith("x86_64-"):
        return Target(
            {
                "kind": "llvm",
                "mtriple": "x86_64-apple-macos",
                "mcpu": host_cpu,
            }
        )
    elif target_triple.startswith("arm64-"):
        return Target(
            {
                "kind": "llvm",
                "mtriple": "arm64-apple-macos",
                "mcpu": host_cpu,
            }
        )
    else:
        raise RuntimeError("Unsupported target triple: %s" % target_triple)


def detect_local_metal():
    dev = tvm.metal()
    if not dev.exist:
        return None

    return Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": 32,
        },
        host=detect_local_metal_host(),
    )


def detect_local_cuda():
    dev = tvm.cuda()
    if not dev.exist:
        return None

    return Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + dev.compute_version.replace(".", ""),
        }
    )


def detect_local_rocm():
    dev = tvm.rocm()
    if not dev.exist:
        return None

    return Target(
        {
            "kind": "rocm",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def detect_local_vulkan():
    dev = tvm.vulkan()
    if not dev.exist:
        return None

    return Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": 1,
            "supports_int16": 1,
            "supports_int8": 1,
            "supports_16bit_buffer": 1,
        }
    )


def detect_local_opencl():
    dev = tvm.opencl()
    if not dev.exist:
        return None

    return Target("opencl")


def detect_local_target() -> Target:
    for method in [
        detect_local_metal,
        detect_local_rocm,
        detect_local_cuda,
        detect_local_vulkan,
        detect_local_opencl,
    ]:
        target = method()
        if target is not None:
            return target

    print("Failed to detect local GPU, falling back to CPU as a target")
    return Target("llvm")


def parse_target(target: str) -> tuple[Target, str]:
    print(target)

    if target == "auto":
        tvm_target = detect_local_target()
        if tvm_target.host is None:
            tvm_target = Target(
                target,
                host="llvm",    # TODO: detect host cpu
            )
        tvm_target_kind = tvm_target.kind.default_keys[0]
    elif target in ["cuda", "cuda-multiarch"]:
        tvm_target = detect_local_cuda()
        if tvm_target is None:
            raise RuntimeError("No local CUDA GPU found!")

        tvm_target_kind = tvm_target.kind.default_keys[0]
        if target == "cuda-multiarch":
            tvm_target_kind += "-multiarch"
    elif target == "metal":
        tvm_target = detect_local_metal()
        if tvm_target is None:
            logger.warning("No local Apple Metal GPU found, Falling back...")
            tvm_target = Target(
                Target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 256,
                        "max_shared_memory_per_block": 32768,
                        "thread_warp_size": 1,
                    }
                ),
                host=detect_local_metal_host(),
            )

        tvm_target_kind = tvm_target.kind.default_keys[0]
    elif target == "llvm":
        tvm_target = Target(target, host="llvm")
        tvm_target_kind = tvm_target.kind.default_keys[0]
    else:
        raise RuntimeError(f"Unsupported target: {target}")

    print(tvm_target, tvm_target_kind)

    if tvm_target_kind == "cuda-multiarch":
        from tvm.contrib import nvcc

        assert tvm_target.arch[3:] != ""
        if int(tvm_target.arch[3:]) >= 70:
            compute_versions = [70, 72, 75, 80, 86, 87, 89, 90]
        else:
            compute_versions = [60, 61, 62]

        tvm_target_kind = "cuda"

        @tvm.register_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
            """use nvcc to generate fatbin code for better optimization"""
            arch = []
            for compute_version in compute_versions:
                arch += ["-gencode", f"arch=compute_{compute_version},code=sm_{compute_version}"]
            ptx = nvcc.compile_cuda(code, target_format="fatbin", arch=arch)
            return ptx

    logger.info(f"Using target: {tvm_target}")

    return tvm_target, tvm_target_kind
