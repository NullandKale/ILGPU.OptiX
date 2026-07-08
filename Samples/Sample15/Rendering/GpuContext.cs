using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime.Cuda;
using System;
using System.Runtime.InteropServices;

namespace Sample15
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
            Context = Context.Create(b => b.Cuda().EnableAlgorithms()
                                           .Optimize(OptimizationLevel.O2).Math(MathMode.Default).Inlining(InliningMode.Aggressive).LibDevice());
            Accelerator = Context.CreateCudaAccelerator(0);

            // Validation mode ALL + a log callback surfaces OptiX's own descriptive
            // error/warning text (e.g. "these build input types cannot be combined in
            // one GAS") instead of a bare OptixResult code with no message - the accel
            // build entry points (unlike module/pipeline creation) never return a log
            // string of their own, so this is the only way to get that detail out of
            // the driver.
            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            RayTracer = OptixRayTracer.Create(Accelerator, new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
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
