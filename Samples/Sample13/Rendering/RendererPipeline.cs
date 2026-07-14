using ILGPU.OptiX;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.AccelStructures;
using System;

namespace Sample13
{
    /// <summary>
    /// Compiles every OptiX program into kernels/program groups via
    /// <see cref="OptixRayTracer.CreatePipeline{TLaunchParams}"/> and links the
    /// pipeline. Built once at startup; only the hitgroup records vary per scene (see
    /// <see cref="SbtBuilder"/>).
    ///
    /// Nine named hit groups per ray type (see <see cref="HitGroupKinds"/>) - one
    /// "triangle" group plus one per custom-primitive kind plus "volumeGrid" - all
    /// sharing the same closest-hit/any-hit programs within a ray type and differing
    /// only by intersection program, exactly matching the sample's previous
    /// CreateCustomHitgroupKernel closure. RayTypeBuilder.HitGroup(kind, ...) compiles a distinct program group
    /// per call even when closestHit/anyHit are the same delegate reference.
    /// </summary>
    public sealed class RendererPipeline : IDisposable
    {
        public RayTracingPipeline<LaunchParams> Pipeline { get; }

        public RendererPipeline(GpuContext gpu)
        {
            Pipeline = gpu.RayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenProgram.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<Payloads.RadiancePayload>()
                    .Miss(MissAndShadowPrograms.__miss__radiance)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Triangle, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Sphere, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__sphere)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Box, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__box)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CylinderY, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__cylinderY)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Disk, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__disk)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.XYRect, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__xyRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.XZRect, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__xzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.YZRect, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__yzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.VolumeGrid, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__volumeGrid))
                .RayType("shadow", r => r
                    .Payload<Payloads.ShadowPayload>()
                    .Miss(MissAndShadowPrograms.__miss__shadow)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Triangle, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Sphere, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__sphere)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Box, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__box)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CylinderY, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__cylinderY)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Disk, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__disk)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.XYRect, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__xyRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.XZRect, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__xzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.YZRect, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__yzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.VolumeGrid, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__volumeGrid))
                // Triangle and custom-primitive geometry cannot be combined as multiple
                // build inputs within a single GAS (confirmed against the OptiX SDK's
                // own optixSimpleMotionBlur sample, which builds one GAS per
                // build-input type and combines them via an IAS) - so this is a single
                // level of instancing (one IAS directly over two GASes), not a single
                // GAS, hence the non-default traversable graph flag.
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
        }

        public void Dispose() => Pipeline.Dispose();
    }

    /// <summary>
    /// Named hit-group kinds shared between <see cref="RendererPipeline"/> (which
    /// declares one <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(string, System.Action{TLaunchParams}, System.Action{TLaunchParams}, System.Action{TLaunchParams})"/>
    /// per kind on each ray type) and <see cref="SbtBuilder"/> (which tags each
    /// <see cref="HitGroupEntry{TMaterial}"/> it packs with the matching kind) - one
    /// source of truth so the two can never drift out of sync on spelling.
    /// </summary>
    public static class HitGroupKinds
    {
        public const string Triangle = "triangle";
        public const string Sphere = "sphere";
        public const string Box = "box";
        public const string CylinderY = "cylinderY";
        public const string Disk = "disk";
        public const string XYRect = "xyRect";
        public const string XZRect = "xzRect";
        public const string YZRect = "yzRect";
        public const string VolumeGrid = "volumeGrid";
    }
}
