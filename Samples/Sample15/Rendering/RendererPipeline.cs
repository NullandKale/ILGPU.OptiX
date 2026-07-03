using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
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
        // Same closest-hit/any-hit entry points as the triangle hitgroup kernels above,
        // but with a per-custom-primitive-kind __intersection__ program attached -
        // CreateHitgroupKernel bundles module compilation with program-group creation,
        // so a distinct intersection module means a distinct OptixKernel/program group
        // even though the shading code is identical (docs/SAMPLE13_PLAN.md's
        // custom-primitive design). Indexed by IntersectionPrograms.HitKind* (0=Sphere,
        // 1=Box, 2=CylinderY, 3=Disk, 4=XYRect, 5=XZRect, 6=YZRect).
        public OptixKernel[] RadianceHitgroupKernelsCustom { get; } = new OptixKernel[7];
        public OptixKernel[] ShadowHitgroupKernelsCustom { get; } = new OptixKernel[7];
        // The volume grid is one GAS primitive whose hitKind range (7-12) encodes which
        // face was hit rather than "which kind" (see IntersectionPrograms.cs), so it
        // gets one dedicated kernel pair rather than a slot in the arrays above.
        public OptixKernel RadianceHitgroupKernelVolumeGrid { get; }
        public OptixKernel ShadowHitgroupKernelVolumeGrid { get; }

        public MemoryBuffer1D<RaygenRecord, Stride1D.Dense> RaygenRecordsBuffer { get; }
        public MemoryBuffer1D<MissRecord, Stride1D.Dense> MissRecordsBuffer { get; }

        readonly OptixKernel raygenKernel;
        readonly OptixKernel radianceMissKernel;
        readonly OptixKernel shadowMissKernel;

        public RendererPipeline(GpuContext gpu)
        {
            var deviceContext = gpu.DeviceContext;

            var moduleCompileOptions = new OptixModuleCompileOptions()
            {
                MaxRegisterCount = 50,
                OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE
            };

            var pipelineCompileOptions = new OptixPipelineCompileOptions()
            {
                // Triangle and custom-primitive geometry cannot be combined as multiple
                // build inputs within a single GAS (confirmed against the OptiX SDK's
                // own optixSimpleMotionBlur sample, which builds one GAS per build-input
                // type and combines them via an IAS) - so this is a single level of
                // instancing (one IAS directly over two GASes), not a single GAS.
                TraversableGraphFlags = OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
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
            };

            var pipelineLinkOptions = new OptixPipelineLinkOptions()
            {
                MaxTraceDepth = 2
            };

            raygenKernel = deviceContext.CreateRaygenKernel<LaunchParams>(
                RaygenProgram.__raygen__renderFrame,
                moduleCompileOptions,
                pipelineCompileOptions);

            radianceMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                MissAndShadowPrograms.__miss__radiance,
                moduleCompileOptions,
                pipelineCompileOptions);

            shadowMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                MissAndShadowPrograms.__miss__shadow,
                moduleCompileOptions,
                pipelineCompileOptions);

            RadianceHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                ClosestHitProgram.__closest__radiance,
                MissAndShadowPrograms.__anyhit__radiance,
                null,
                moduleCompileOptions,
                pipelineCompileOptions);

            ShadowHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                MissAndShadowPrograms.__closesthit__shadow,
                MissAndShadowPrograms.__anyhit__shadow,
                null,
                moduleCompileOptions,
                pipelineCompileOptions);

            void CreateCustomHitgroupKernel(uint kind, Action<LaunchParams> intersectionKernel)
            {
                RadianceHitgroupKernelsCustom[kind] = deviceContext.CreateHitgroupKernel<LaunchParams>(
                    ClosestHitProgram.__closest__radiance,
                    MissAndShadowPrograms.__anyhit__radiance,
                    intersectionKernel,
                    moduleCompileOptions,
                    pipelineCompileOptions);

                ShadowHitgroupKernelsCustom[kind] = deviceContext.CreateHitgroupKernel<LaunchParams>(
                    MissAndShadowPrograms.__closesthit__shadow,
                    MissAndShadowPrograms.__anyhit__shadow,
                    intersectionKernel,
                    moduleCompileOptions,
                    pipelineCompileOptions);
            }

            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindSphere, IntersectionPrograms.__intersection__sphere);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindBox, IntersectionPrograms.__intersection__box);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindCylinderY, IntersectionPrograms.__intersection__cylinderY);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindDisk, IntersectionPrograms.__intersection__disk);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindXYRect, IntersectionPrograms.__intersection__xyRect);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindXZRect, IntersectionPrograms.__intersection__xzRect);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindYZRect, IntersectionPrograms.__intersection__yzRect);

            RadianceHitgroupKernelVolumeGrid = deviceContext.CreateHitgroupKernel<LaunchParams>(
                ClosestHitProgram.__closest__radiance,
                MissAndShadowPrograms.__anyhit__radiance,
                IntersectionPrograms.__intersection__volumeGrid,
                moduleCompileOptions,
                pipelineCompileOptions);

            ShadowHitgroupKernelVolumeGrid = deviceContext.CreateHitgroupKernel<LaunchParams>(
                MissAndShadowPrograms.__closesthit__shadow,
                MissAndShadowPrograms.__anyhit__shadow,
                IntersectionPrograms.__intersection__volumeGrid,
                moduleCompileOptions,
                pipelineCompileOptions);

            var raygenKernels = new[] { raygenKernel };
            var missKernels = new[] { radianceMissKernel, shadowMissKernel };
            var allKernels = raygenKernels.Concat(missKernels)
                .Concat(new[] { RadianceHitgroupKernel, ShadowHitgroupKernel, RadianceHitgroupKernelVolumeGrid, ShadowHitgroupKernelVolumeGrid })
                .Concat(RadianceHitgroupKernelsCustom)
                .Concat(ShadowHitgroupKernelsCustom)
                .ToArray();

            Pipeline = deviceContext.CreatePipeline(
                pipelineCompileOptions,
                pipelineLinkOptions,
                allKernels.Select(x => x.ProgramGroup).ToArray());

            // maxTraversableGraphDepth = 2: our traversable graph is one IAS directly
            // over GASes (triangles/custom primitives), not a single bare GAS - per
            // optix_host.h's optixPipelineSetStackSize docs, "for a simple IAS -> GAS
            // traversal graph, the maxTraversableGraphDepth is two".
            Pipeline.SetStackSize(2 * 1024, 2 * 1024, 2 * 1024, 2);

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

            ShadowHitgroupKernelVolumeGrid.Dispose();
            RadianceHitgroupKernelVolumeGrid.Dispose();
            foreach (var kernel in RadianceHitgroupKernelsCustom)
                kernel.Dispose();
            foreach (var kernel in ShadowHitgroupKernelsCustom)
                kernel.Dispose();

            ShadowHitgroupKernel.Dispose();
            RadianceHitgroupKernel.Dispose();
            shadowMissKernel.Dispose();
            radianceMissKernel.Dispose();
            raygenKernel.Dispose();
        }
    }
}
