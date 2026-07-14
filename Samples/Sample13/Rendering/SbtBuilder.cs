using ILGPU.OptiX.Pipeline;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System.Collections.Generic;

namespace Sample13
{
    // The scene->hitgroup-record mapping and everything that computes with it live
    // together in this file on purpose: SbtBuilder.Build's record layout and the IAS
    // SbtOffset math (via SbtLayout, also used by AccelStructureBuilder) are one
    // invariant - change one and the others must follow.

    /// <summary>
    /// The single source of truth for how a scene's triangles map onto GAS build
    /// inputs and hitgroup SBT record ranges - shared by <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, and the renderer's HUD summary. Thin
    /// sample-specific adapter over <see cref="ILGPU.OptiX.Pipeline.OptixSbtLayout"/> -
    /// the actual range-resolution logic used to be copy-pasted identically into this
    /// file by Sample13/14/15; it's now defined once in the shared library.
    /// </summary>
    public static class SbtLayout
    {
        public static MeshRange[] GetTriangleMeshRanges(SceneData scene, bool useMergedTrianglesGas) =>
            ILGPU.OptiX.Pipeline.OptixSbtLayout.GetTriangleMeshRanges(
                scene.Indices.Length, scene.MeshRanges, useMergedTrianglesGas);
    }

    /// <summary>
    /// Builds the per-scene hitgroup portion of the shader binding table via
    /// <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(System.ReadOnlySpan{HitGroupEntry{TMaterial}}, int)"/>.
    ///
    /// Hitgroup entries are built as [triangle mat0, mat1, ...] repeated per triangle
    /// mesh build input, followed by the same per-material sequence again for each
    /// custom-primitive kind actually present in the scene, in canonical kind order
    /// (Sphere, Box, CylinderY, Disk, XYRect, XZRect, YZRect - matching
    /// IntersectionPrograms.HitKind* and the custom-primitives GAS's build-input
    /// order), then one material-agnostic volume-grid entry if present. Each entry is
    /// tagged with the matching <see cref="HitGroupKinds"/> name, and
    /// SetHitRecords interleaves it across both ray types (radiance, shadow)
    /// internally - this relies on OptiX automatically summing NumSbtRecords across
    /// build inputs within a GAS to compute each build input's base SBT-GAS-index, so
    /// each build input's own SbtIndexOffsetBuffer values stay local/0-based.
    /// </summary>
    public sealed class SbtBuilder
    {
        readonly RendererPipeline pipeline;
        readonly TextureCache textures;

        public SbtBuilder(RendererPipeline pipeline, TextureCache textures)
        {
            this.pipeline = pipeline;
            this.textures = textures;
        }

        public void Build(SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            // One full Materials.Length block per triangle mesh build input (see
            // SbtLayout.GetTriangleMeshRanges), not just one for the whole scene -
            // mirrors the per-custom-primitive-kind blocks below.
            int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;

            (int Count, string Kind)[] customKinds =
            {
                (scene.Spheres.Length, HitGroupKinds.Sphere),
                (scene.Boxes.Length, HitGroupKinds.Box),
                (scene.CylindersY.Length, HitGroupKinds.CylinderY),
                (scene.Disks.Length, HitGroupKinds.Disk),
                (scene.XYRects.Length, HitGroupKinds.XYRect),
                (scene.XZRects.Length, HitGroupKinds.XZRect),
                (scene.YZRects.Length, HitGroupKinds.YZRect),
            };

            // Resolved once per material index - any material without a texture path
            // (the overwhelming majority of Sample13's scenes) keeps TextureObject=0,
            // matching Sample08's "0 = no texture" convention.
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

            foreach (var (count, kind) in customKinds)
            {
                if (count == 0)
                    continue;
                for (var i = 0; i < scene.Materials.Length; i++)
                    entries.Add(new HitGroupEntry<MaterialSbtData>(kind, BuildMaterialRecord(i)));
            }

            // NumSbtRecords=1 for this build input (see the custom-primitives GAS
            // build) - its record's own custom data is never read (ShadeVolumeGrid
            // looks up materials directly via LaunchParams.Materials instead), so a
            // single default-valued entry (for the kernel/program-group dispatch)
            // suffices.
            bool hasVolumeGrid = scene.VoxelMaterialIds.Length > 0;
            if (hasVolumeGrid)
                entries.Add(new HitGroupEntry<MaterialSbtData>(HitGroupKinds.VolumeGrid, default));

            pipeline.Pipeline.SetHitRecords<MaterialSbtData>(entries.ToArray());
        }
    }
}
