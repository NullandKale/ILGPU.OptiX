// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: CudaKernelDiagnostics.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Cuda
{
    // Queries a compiled CUDA kernel's driver-JIT'd register/local-memory usage via
    // cuFuncGetAttribute - neither ILGPU nor ILGPU.OptiX wrap this entry point (grepped
    // ILGPU's own CudaAPI.cs/CudaNativeMethods.cs - no CUfunction_attribute binding
    // exists), so this P/Invokes nvcuda.dll directly, same pattern as
    // CudaTextureObject.cs's own doc comment explains.
    //
    // ONLY applicable to plain ILGPU-launched kernels (TonemapKernel, TaaResolveKernel,
    // AdamKernel, ...) that go through Accelerator.LoadAutoGroupedKernel/LoadKernel and
    // therefore have a real CUfunction. OptiX raygen/closesthit/miss/anyhit programs do
    // NOT - they're compiled to PTX and invoked via optixLaunch against a pipeline+SBT,
    // never through cuLaunchKernel/CUfunction, so there is no register count to query
    // for them through this (or any) CUDA driver API call.
    //
    // Usage: the convenience extensions (LoadAutoGroupedStreamKernel&lt;...&gt;) return an
    // opaque launcher delegate with no accessible Kernel/CudaKernel handle - get one
    // instead via the lower-level Accelerator.LoadAutoGroupedKernel(MethodInfo) or
    // LoadKernel(MethodInfo) overload, which returns an ILGPU.Runtime.Kernel that is a
    // CudaKernel on a CudaAccelerator:
    //   var kernel = (CudaKernel)accelerator.LoadAutoGroupedKernel(
    //       typeof(MyKernels).GetMethod(nameof(MyKernels.MyMethod)));
    //   int regs = CudaKernelDiagnostics.GetRegisterCount(kernel);
    [CLSCompliant(false)]
    public static class CudaKernelDiagnostics
    {
        // CUfunction_attribute (cuda.h) - only the subset relevant to an occupancy/
        // register-pressure investigation is exposed; the full enum has a few more
        // (PTX_VERSION, BINARY_VERSION, CACHE_MODE_CA, ...) not worth a public surface
        // here until something actually needs them.
        private enum CuFunctionAttribute
        {
            MaxThreadsPerBlock = 0,
            SharedSizeBytes = 1,
            ConstSizeBytes = 2,
            LocalSizeBytes = 3,
            NumRegs = 4,
        }

        [DllImport("nvcuda.dll")]
        private static extern int cuFuncGetAttribute(out int pi, int attrib, IntPtr hfunc);

        /// <summary>
        /// Registers used per thread, as JIT-compiled by the CUDA driver for the current
        /// GPU - the direct occupancy-limiting number (SM register-file size / this =
        /// max resident threads from the register constraint alone, before shared-memory/
        /// block-size limits are applied on top).
        /// </summary>
        public static int GetRegisterCount(CudaKernel kernel) =>
            GetAttribute(kernel, CuFunctionAttribute.NumRegs);

        /// <summary>
        /// Bytes of local memory (register-spill scratch, plus any explicit local
        /// arrays) used per thread. Nonzero means the compiler could not keep every
        /// live value in registers and spilled some to (slow, per-thread) local memory -
        /// a strong signal the kernel's live-variable footprint exceeds what fits
        /// comfortably in the register file, independent of the raw register count
        /// above (a kernel can show a moderate register count and still be spilling
        /// heavily if the compiler traded registers for spills to hit a target
        /// occupancy).
        /// </summary>
        public static int GetLocalMemoryBytes(CudaKernel kernel) =>
            GetAttribute(kernel, CuFunctionAttribute.LocalSizeBytes);

        /// <summary>Static shared memory used per block, in bytes.</summary>
        public static int GetSharedMemoryBytes(CudaKernel kernel) =>
            GetAttribute(kernel, CuFunctionAttribute.SharedSizeBytes);

        /// <summary>The driver-reported maximum block size this kernel can be launched with, given its register/shared-memory footprint.</summary>
        public static int GetMaxThreadsPerBlock(CudaKernel kernel) =>
            GetAttribute(kernel, CuFunctionAttribute.MaxThreadsPerBlock);

        private static int GetAttribute(CudaKernel kernel, CuFunctionAttribute attribute)
        {
            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));
            CudaCheck(cuFuncGetAttribute(out int value, (int)attribute, kernel.FunctionPtr));
            return value;
        }

        private static void CudaCheck(int cuResult)
        {
            if (cuResult != 0)
                throw new InvalidOperationException($"CUDA driver call failed with CUresult {cuResult}.");
        }
    }
}
