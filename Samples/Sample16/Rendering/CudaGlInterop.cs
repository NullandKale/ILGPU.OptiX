using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Cuda;
using System;
using System.Runtime.InteropServices;

namespace Sample16
{
    // Same three-flag enum and P/Invoke signature set as Sample13's
    // Rendering/CudaGlInterop.cs (itself copied from
    // example/OpenTKSplat/OpenTKSplat/Compute/ILGPUOpenGLExchangeBuffer.cs) - a
    // proven, minimal wrapper around the CUDA driver's GL-interop entry points
    // (nvcuda.dll, same P/Invoke convention CudaTextureObject.cs already uses
    // elsewhere in this repo for texture objects).
    public enum CudaGraphicsMapFlags
    {
        None = 0,
        ReadOnly = 1,
        WriteDiscard = 2
    }

    public static class CudaGlInterop
    {
        [DllImport("nvcuda", EntryPoint = "cuGraphicsGLRegisterBuffer")]
        public static extern CudaError RegisterBuffer(
            out IntPtr resource,
            int buffer,
            uint flags);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsMapResources")]
        public static extern CudaError MapResources(
            int count,
            IntPtr resources,
            IntPtr stream);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsUnmapResources")]
        public static extern CudaError UnmapResources(
            int count,
            IntPtr resources,
            IntPtr stream);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsResourceGetMappedPointer_v2")]
        public static extern CudaError GetMappedPointer(
            out IntPtr devicePtr,
            out int size,
            IntPtr resource);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsUnregisterResource")]
        public static extern CudaError UnregisterResource(IntPtr resource);
    }
}
