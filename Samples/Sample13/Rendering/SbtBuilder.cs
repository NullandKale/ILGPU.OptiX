using ILGPU.OptiX.Pipeline;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System.Collections.Generic;

namespace Sample13
{
    // The hit-group kind names plus everything that computes with the hitgroup record
    // ordering live together in this file on purpose: SbtBuilder.Apply's entry order,
    // the RayType(...).HitGroup(kind, ...) declarations in SampleRenderer's
    // CreatePipeline(...) call, and the IAS SbtOffset math (via SbtLayout, also used by
    // AccelStructureBuilder) are one invariant - change one and the others must follow.

    /// <summary>
    /// The single source of truth for how a scene's triangles map onto GAS build
    /// inputs and hitgroup SBT record ranges - shared by <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, and the renderer's HUD summary. Thin
    /// sample-specific adapter over <see cref="ILGPU.OptiX.Pipeline.OptixSbtLayout"/>,
    /// which owns the actual range-resolution logic.
    /// </summary>
    public static class SbtLayout
    {
        public static MeshRange[] GetTriangleMeshRanges(SceneData scene, bool useMergedTrianglesGas) =>
            ILGPU.OptiX.Pipeline.OptixSbtLayout.GetTriangleMeshRanges(
                scene.Indices.Length, scene.MeshRanges, useMergedTrianglesGas);
    }

    /// <summary>
    /// Hit-group kind names registered on every ray type in
    /// <see cref="SampleRenderer"/>'s <c>CreatePipeline(...)</c> call - one per
    /// intersection program sharing that ray type's closest-hit/any-hit pair (see
    /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(string, System.Action{TLaunchParams}, System.Action{TLaunchParams}?, System.Action{TLaunchParams}?)"/>).
    /// Indexed by <see cref="IntersectionPrograms.HitKind*"/> for the 7 custom-primitive
    /// kinds; <see cref="Triangle"/> and <see cref="VolumeGrid"/> stand alone.
    /// </summary>
    public static class HitGroupKinds
    {
        public const string Triangle = "triangle";
        public const string VolumeGrid = "volumeGrid";

        // Canonical kind order matching IntersectionPrograms.HitKind* (0=Sphere,
        // 1=Box, 2=CylinderY, 3=Disk, 4=XYRect, 5=XZRect, 6=YZRect) and the
        // custom-primitives GAS's build-input order.
        public static readonly string[] CustomPrimitiveKinds =
        {
            "sphere", "box", "cylinderY", "disk", "xyRect", "xzRect", "yzRect",
        };
    }

    /// <summary>
    /// Builds the per-scene hitgroup portion of the shader binding table via
    /// <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(System.ReadOnlySpan{HitGroupEntry{TMaterial}}, int)"/>.
    ///
    /// Hitgroup entries are laid out as [triangle mat0, mat1, ...] repeated per
    /// triangle mesh build input, followed by the same per-material sequence again for
    /// each custom-primitive kind actually present in the scene, in canonical kind
    /// order (Sphere, Box, CylinderY, Disk, XYRect, XZRect, YZRect - matching
    /// IntersectionPrograms.HitKind* and the custom-primitives GAS's build-input
    /// order), followed by one volume-grid entry if present. Each entry expands into
    /// one record per ray type (radiance, shadow) - see SetHitRecords's own doc
    /// comment. This relies on OptiX automatically summing NumSbtRecords across build
    /// inputs within a GAS to compute each build input's base SBT-GAS-index, so each
    /// build input's own SbtIndexOffsetBuffer values stay local/0-based.
    /// </summary>
    public sealed class SbtBuilder
    {
        readonly TextureCache textures;

        public SbtBuilder(TextureCache textures)
        {
            this.textures = textures;
        }

        public void Apply(RayTracingPipeline<LaunchParams> pipeline, SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            // One full Materials.Length block per triangle mesh build input (see
            // SbtLayout.GetTriangleMeshRanges), not just one for the whole scene -
            // mirrors the per-custom-primitive-kind blocks below.
            int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;
            int[] customCounts =
            {
                scene.Spheres.Length, scene.Boxes.Length, scene.CylindersY.Length, scene.Disks.Length,
                scene.XYRects.Length, scene.XZRects.Length, scene.YZRects.Length,
            };
            bool hasVolumeGrid = scene.VoxelMaterialIds.Length > 0;

            // Resolved once per material index - any material without a texture path
            // (the overwhelming majority of this sample's scenes) keeps
            // TextureObject=0, meaning "no texture".
            var textureHandles = new ulong[scene.Materials.Length];
            for (var i = 0; i < scene.Materials.Length; i++)
            {
                string path = i < scene.MaterialTexturePaths.Length ? scene.MaterialTexturePaths[i] : null;
                textureHandles[i] = string.IsNullOrEmpty(path) ? 0 : textures.GetOrLoad(path);
            }

            MaterialSbtData BuildMaterialRecord(int i)
            {
                var mat = scene.Materials[i];
                return new MaterialSbtData
                {
                    Albedo = mat.Albedo,
                    Reflectivity = mat.Reflectivity,
                    Emission = mat.Emission,
                    Transparency = mat.Transparency,
                    IndexOfRefraction = mat.IndexOfRefraction,
                    TransmissionColor = mat.TransmissionColor,
                    TextureObject = textureHandles[i],
                    TextureWeight = mat.TextureWeight,
                    UVScale = mat.UVScale,
                    MaterialKind = mat.MaterialKind,
                    CheckerColorB = mat.CheckerColorB,
                    CheckerScale = mat.CheckerScale,
                    AlphaCutoff = mat.AlphaCutoff,
                };
            }

            var entries = new List<HitGroupEntry<MaterialSbtData>>();
            for (var m = 0; m < triangleMeshCount; m++)
            {
                for (var i = 0; i < scene.Materials.Length; i++)
                    entries.Add(new HitGroupEntry<MaterialSbtData>(HitGroupKinds.Triangle, BuildMaterialRecord(i)));
            }
            for (var kind = 0; kind < customCounts.Length; kind++)
            {
                if (customCounts[kind] == 0)
                    continue;
                for (var i = 0; i < scene.Materials.Length; i++)
                    entries.Add(new HitGroupEntry<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[kind], BuildMaterialRecord(i)));
            }
            if (hasVolumeGrid)
            {
                // Its record's own custom data is never read (ShadeVolumeGrid looks up
                // materials directly via LaunchParams.Materials instead), so only the
                // kind (for header/program-group dispatch) matters here.
                entries.Add(new HitGroupEntry<MaterialSbtData>(HitGroupKinds.VolumeGrid, default));
            }

            pipeline.SetHitRecords<MaterialSbtData>(entries.ToArray());
        }
    }
}
