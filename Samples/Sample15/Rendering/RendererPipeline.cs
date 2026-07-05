using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using System;
using System.Linq;

namespace Sample15
{
    /// <summary>
    /// Compiles every OptiX program into kernels/program groups, links the pipeline,
    /// and packs the (scene-independent) raygen/miss SBT records. Built once at
    /// startup; only the hitgroup records vary per scene (see <see cref="SbtBuilder"/>).
    /// </summary>
    public sealed class RendererPipeline : IDisposable
    {
        public OptixPipeline Pipeline { get; }

        public OptixKernel RadianceHitgroupKernel { get; }
        public OptixKernel ShadowHitgroupKernel { get; }

        public MemoryBuffer1D<RaygenRecord, Stride1D.Dense> RaygenRecordsBuffer { get; }
        public MemoryBuffer1D<MissRecord, Stride1D.Dense> MissRecordsBuffer { get; }

        readonly OptixKernel raygenKernel;
        readonly OptixKernel radianceMissKernel;
        readonly OptixKernel shadowMissKernel;

        public RendererPipeline(GpuContext gpu)
        {
            var deviceContext = gpu.DeviceContext
                .WithModuleCompileOptions(new OptixModuleCompileOptions()
                {
                    MaxRegisterCount = 50,
                    OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                    DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE
                })
                .WithPipelineCompileOptions(new OptixPipelineCompileOptions()
                {
                    // A single bare triangles GAS, no IAS (custom primitives/volume grid
                    // were removed - every scene is now pure triangle geometry).
                    TraversableGraphFlags = OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                    // Sourced from Payloads.RadiancePayloadCount rather than a separate
                    // literal here - color(3) + flag(1) + new-ray-origin(3) +
                    // new-ray-direction(3) + throughput-tint(3) + normal(3) + albedo(3) +
                    // carried RNG state(1) = 20, see Payloads.cs's SetContinuePayload/
                    // SetTerminalPayload/SetAovPayload/GetCarriedRngState and
                    // docs/SAMPLE15_PLAN.md Milestone M3. The shadow ray's own
                    // transmittance(3)+hitCount(1) = 4 payloads fit within this.
                    NumPayloadValues = Payloads.RadiancePayloadCount,
                    NumAttributeValues = 2,
                    ExceptionFlags = OptixExceptionFlags.OPTIX_EXCEPTION_FLAG_NONE,
                    PipelineLaunchParamsVariableName = OptixLaunchParams.VariableName
                })
                .WithPipelineLinkOptions(new OptixPipelineLinkOptions()
                {
                    MaxTraceDepth = 2
                });

            raygenKernel = deviceContext.CreateRaygenKernel<LaunchParams>(
                RaygenProgram.__raygen__renderFrame);

            radianceMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                MissAndShadowPrograms.__miss__radiance);

            shadowMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                MissAndShadowPrograms.__miss__shadow);

            RadianceHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                ClosestHitProgram.__closest__radiance,
                MissAndShadowPrograms.__anyhit__radiance,
                null);

            ShadowHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                MissAndShadowPrograms.__closesthit__shadow,
                MissAndShadowPrograms.__anyhit__shadow,
                null);

            var raygenKernels = new[] { raygenKernel };
            var missKernels = new[] { radianceMissKernel, shadowMissKernel };

            // Build pipeline using builder
            var pipelineBuilder = new OptixPipelineBuilder();
            pipelineBuilder.AddKernels(raygenKernels);
            pipelineBuilder.AddKernels(missKernels);
            pipelineBuilder.AddKernels(new[] { RadianceHitgroupKernel, ShadowHitgroupKernel });
            Pipeline = pipelineBuilder.Build(deviceContext);

            // maxTraversableGraphDepth = 1: a single bare GAS, no IAS indirection.
            Pipeline.SetStackSize(2 * 1024, 2 * 1024, 2 * 1024, 1);

            var raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            var missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            RaygenRecordsBuffer = gpu.Accelerator.Allocate1D(raygenRecordsArray);
            MissRecordsBuffer = gpu.Accelerator.Allocate1D(missRecordsArray);
        }

        public void Dispose()
        {
            MissRecordsBuffer.Dispose();
            RaygenRecordsBuffer.Dispose();

            Pipeline.Dispose();

            ShadowHitgroupKernel.Dispose();
            RadianceHitgroupKernel.Dispose();
            shadowMissKernel.Dispose();
            radianceMissKernel.Dispose();
            raygenKernel.Dispose();
        }
    }
}
