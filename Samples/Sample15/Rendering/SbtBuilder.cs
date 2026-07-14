using ILGPU.OptiX.Pipeline;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;

namespace Sample15
{
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
    /// Builds the per-scene hitgroup portion of the shader binding table and uploads it
    /// via <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(MaterialSbtData[][], int)"/>.
    ///
    /// Hitgroup records are laid out as [triangle mat0-radiance, mat0-shadow, mat1-
    /// radiance, mat1-shadow, ...] repeated per triangle mesh build input (the
    /// <c>repeatCount</c> argument below). This relies on OptiX automatically summing
    /// NumSbtRecords across build inputs within a GAS to compute each build input's
    /// base SBT-GAS-index, so each build input's own SbtIndexOffsetBuffer values stay
    /// local/0-based.
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
            // One full Materials.Length radiance+shadow block per triangle mesh build
            // input (see SbtLayout.GetTriangleMeshRanges).
            int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;

            var radianceMaterials = new MaterialSbtData[scene.Materials.Length];

            // The shadow ray type's hitgroup records stay default/zeroed - matches
            // the pre-existing "only the radiance record's data is ever read" contract
            // (see MaterialSbtData.cs's doc comment; the shadow closest-hit program is
            // an empty stub).
            var shadowMaterials = new MaterialSbtData[scene.Materials.Length];

            // Resolved once per material index - any material without a texture path
            // (the overwhelming majority of Sample13's scenes) keeps TextureObject=0,
            // matching Sample08's "0 = no texture" convention. NormalTexture/OrmTexture
            // follow the exact same resolution as
            // BaseColorTexture - MaterialSbtData.OrmTexture/NormalTexture themselves are
            // never read here; only the parallel *TexturePaths arrays are.
            for (var i = 0; i < scene.Materials.Length; i++)
            {
                var mat = scene.Materials[i];

                string path = i < scene.MaterialTexturePaths.Length ? scene.MaterialTexturePaths[i] : null;
                ulong baseColorTexture = string.IsNullOrEmpty(path) ? 0 : textures.GetOrLoad(path);

                string normalPath = i < scene.MaterialNormalTexturePaths.Length ? scene.MaterialNormalTexturePaths[i] : null;
                ulong normalTexture = string.IsNullOrEmpty(normalPath) ? 0 : textures.GetOrLoad(normalPath);

                string ormPath = i < scene.MaterialOrmTexturePaths.Length ? scene.MaterialOrmTexturePaths[i] : null;
                ulong ormTexture = string.IsNullOrEmpty(ormPath) ? 0 : textures.GetOrLoad(ormPath);

                ref var record = ref radianceMaterials[i];
                record.BaseColor = mat.BaseColor;
                record.Metallic = mat.Metallic;
                record.Roughness = mat.Roughness;
                record.IOR = mat.IOR;
                record.Emission = mat.Emission;
                record.EmissionStrength = mat.EmissionStrength;
                record.Transmission = mat.Transmission;
                record.TransmissionColor = mat.TransmissionColor;
                record.TransmissionRoughness = mat.TransmissionRoughness;
                record.BaseColorTexture = baseColorTexture;
                record.OrmTexture = ormTexture;
                record.NormalTexture = normalTexture;
                record.NormalStrength = mat.NormalStrength;
                record.TextureWeight = mat.TextureWeight;
                record.UVScale = mat.UVScale;
                record.MaterialKind = mat.MaterialKind;
                record.CheckerColorB = mat.CheckerColorB;
                record.CheckerScale = mat.CheckerScale;
                record.AlphaCutoff = mat.AlphaCutoff;
                record.EmissiveLightListBase = mat.EmissiveLightListBase;
            }

            // Ray type declaration order in SampleRenderer's CreatePipeline(...) call
            // is ["radiance", "shadow"] - this array order must match.
            pipeline.SetHitRecords(new[] { radianceMaterials, shadowMaterials }, triangleMeshCount);
        }
    }
}
