using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace Sample21
{
    /// <summary>
    /// Owns the process-lifetime GPU objects: the ILGPU context/accelerator and the
    /// OptiX ray tracer (device context, with validation + log callback). Created
    /// first, disposed last.
    /// </summary>
    public sealed class GpuContext : IDisposable
    {
        public Context Context { get; }
        public CudaAccelerator Accelerator { get; }
        public OptixRayTracer RayTracer { get; }

        /// <summary>
        /// Escape hatch for consumers that still talk to the raw device context
        /// directly (acceleration structures, the denoiser) - <see cref="RayTracer"/>
        /// only wraps pipeline/SBT/launch, not every OptiX surface.
        /// </summary>
        public OptixDeviceContext DeviceContext => RayTracer.DeviceContext;

        // accelerator.DefaultStream is declared as the base AcceleratorStream type; the
        // denoiser API needs the raw CUstream pointer, which only CudaStream exposes.
        public IntPtr DefaultStreamPtr => ((CudaStream)Accelerator.DefaultStream).StreamPtr;

        // Kept as a field (not a local in the ctor) since the GC has no visibility into
        // the native function pointer OptiX holds onto.
        readonly OptixLogCallback logCallback;

        public GpuContext()
        {
            // O2 + fast math + aggressive inlining - every transcendental call in this
            // sample's kernels (GGX sampling, sRGB decode, equirect env-map mapping)
            // only needs the precision fast-math trades away, not the exactness, and
            // the visual result is unchanged at this sample's tolerances.
            // Profiling() enables AcceleratorStream.AddProfilingMarker (CUDA events) -
            // per-stage frame timing (SampleRenderer.render) measures GPU spans with
            // those instead of stopwatch+Synchronize pairs, so the frame only
            // hard-syncs once (FrameOutput.TonemapToDisplay, before the GL unmap).
            // The flag itself adds no per-kernel overhead; markers cost one CUDA
            // event each, only where actually placed.
            Context = Context.Create(b => b.Cuda().EnableAlgorithms().LibDevice().Profiling()
                                           .Optimize(OptimizationLevel.O2).Math(MathMode.Fast32BitOnly).Inlining(InliningMode.Aggressive));
            Accelerator = Context.CreateCudaAccelerator(0);

            // The log callback surfaces OptiX's own descriptive error/warning text
            // (e.g. "these build input types cannot be combined in one GAS") instead of
            // a bare OptixResult code with no message - the accel build entry points
            // (unlike module/pipeline creation) never return a log string of their own,
            // so this is the only way to get that detail out of the driver.
            //
            // Validation mode is DEBUG-ONLY: OptixDeviceContextValidationMode.All
            // injects per-launch checks and synchronization into every optixLaunch -
            // NVIDIA's own docs warn of a significant performance cost. Debug builds
            // keep All + verbose (level 4) logging for the descriptive-diagnostics
            // benefit it exists for; Release runs validation off with warnings-and-up
            // (level 3) logging.
            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            RayTracer = OptixRayTracer.Create(Accelerator, new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
#if DEBUG
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.All,
#else
                LogCallbackLevel = 3,
                ValidationMode = OptixDeviceContextValidationMode.Off,
#endif
            });
        }

        public void Dispose()
        {
            RayTracer.Dispose();
            Accelerator.Dispose();
            Context.Dispose();
        }
    }
}
