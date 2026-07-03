using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Sample15
{
    // The three SBT record layouts plus everything that computes with the hitgroup
    // record ordering live together in this file on purpose: HitgroupRecord's field
    // order, SbtBuilder.Build's record layout, and the IAS SbtOffset math (via
    // SbtLayout, also used by AccelStructureBuilder) are one invariant - change one
    // and the others must follow.

    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct RaygenRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public int ObjectID;
    }

    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct MissRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public int ObjectID;
    }

    // Two records per material (radiance then shadow) - custom data layout must match
    // MaterialSbtData.cs exactly (field-for-field, same order), which is what
    // __closest__radiance actually reads via OptixGetSbtDataPointer (that pointer starts
    // right after Header, not at the start of this struct). Only radiance records ever
    // have their custom data read; shadow records' fields stay zeroed. Size=160 is
    // empirically measured (Marshal.SizeOf<MaterialSbtData>()==128 + the 32-byte
    // OPTIX_SBT_RECORD_HEADER_SIZE, already a 16-byte multiple - no extra padding needed),
    // not hand-derived from field-by-field offset arithmetic, since Vec3's own layout
    // (auto-property-backed) plus alignment padding around the ulong texture handles
    // makes that arithmetic error-prone. If MaterialSbtData's field list changes again,
    // re-measure via Marshal.SizeOf<MaterialSbtData>() rather than recomputing by hand.
    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 160)]
    public unsafe struct HitgroupRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public Vec3 BaseColor;
        public float Metallic;
        public float Roughness;
        public float IOR;
        public Vec3 Emission;
        public float EmissionStrength;
        public float Transmission;
        public Vec3 TransmissionColor;
        public float TransmissionRoughness;
        public ulong BaseColorTexture;
        public ulong OrmTexture;
        public ulong NormalTexture;
        public float NormalStrength;
        public float TextureWeight;
        public float UVScale;
        public int MaterialKind;
        public Vec3 CheckerColorB;
        public float CheckerScale;
        public float AlphaCutoff;
        public int EmissiveLightListBase;
    }

    /// <summary>
    /// The single source of truth for how a scene's triangles map onto GAS build
    /// inputs and hitgroup SBT record ranges - shared by <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, and the renderer's HUD summary.
    /// </summary>
    public static class SbtLayout
    {
        // UseMergedTrianglesGas==true (the default) always returns a single range
        // spanning the whole Indices array, regardless of what SceneData.MeshRanges
        // tracked - the original single-build-input behavior. Otherwise, empty/unset
        // SceneData.MeshRanges still means "treat the whole Indices array as a single
        // implicit mesh" (every scene that doesn't explicitly track mesh boundaries -
        // procedural/test/CSG scenes, the single-mesh scenes). Shared by the triangles
        // GAS build, the hitgroup SBT build, and the IAS SbtOffset math so this
        // fallback logic lives in one place.
        public static MeshRange[] GetTriangleMeshRanges(SceneData scene, bool useMergedTrianglesGas) =>
            (!useMergedTrianglesGas && scene.MeshRanges != null && scene.MeshRanges.Length > 0)
                ? scene.MeshRanges
                : new[] { new MeshRange { IndexStart = 0, IndexCount = scene.Indices.Length } };
    }

    /// <summary>
    /// Builds the per-scene hitgroup portion of the shader binding table.
    ///
    /// Hitgroup records are laid out as [triangle mat0-radiance, mat0-shadow, mat1-
    /// radiance, mat1-shadow, ...] repeated per triangle mesh build input, followed by
    /// the same per-material sequence again for each custom-primitive kind actually
    /// present in the scene, in canonical kind order (Sphere, Box, CylinderY, Disk,
    /// XYRect, XZRect, YZRect - matching IntersectionPrograms.HitKind* and the
    /// custom-primitives GAS's build-input order). This relies on OptiX automatically
    /// summing NumSbtRecords across build inputs within a GAS to compute each build
    /// input's base SBT-GAS-index, so each build input's own SbtIndexOffsetBuffer
    /// values stay local/0-based - see docs/SAMPLE13_PLAN.md.
    /// </summary>
    public sealed class SbtBuilder
    {
        readonly CudaAccelerator accelerator;
        readonly RendererPipeline pipeline;
        readonly TextureCache textures;

        MemoryBuffer1D<HitgroupRecord, Stride1D.Dense> hitgroupRecordsBuffer;

        public SbtBuilder(CudaAccelerator accelerator, RendererPipeline pipeline, TextureCache textures)
        {
            this.accelerator = accelerator;
            this.pipeline = pipeline;
            this.textures = textures;

            // Guards the HitgroupRecord/MaterialSbtData layout invariant documented on
            // HitgroupRecord above - if MaterialSbtData's field list changes without
            // re-measuring HitgroupRecord's explicit Size, this catches the drift
            // immediately (mirrors OptixAPI.Init.cs's OptixFunctionTable size assert).
            Debug.Assert(
                Marshal.SizeOf<HitgroupRecord>() == Marshal.SizeOf<MaterialSbtData>() + OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE,
                "HitgroupRecord's declared Size no longer matches MaterialSbtData's " +
                "actual marshaled size plus the SBT record header - re-measure via " +
                "Marshal.SizeOf<MaterialSbtData>() and update HitgroupRecord's Size.");
        }

        public unsafe OptixShaderBindingTable Build(SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            // One full Materials.Length radiance+shadow block per triangle mesh build
            // input (see SbtLayout.GetTriangleMeshRanges), not just one for the whole
            // scene - mirrors the per-custom-primitive-kind blocks below.
            int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;
            int[] customCounts =
            {
                scene.Spheres.Length, scene.Boxes.Length, scene.CylindersY.Length, scene.Disks.Length,
                scene.XYRects.Length, scene.XZRects.Length, scene.YZRects.Length,
            };

            var hitgroupKernelsList = new List<OptixKernel>();
            for (var m = 0; m < triangleMeshCount; m++)
            {
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    hitgroupKernelsList.Add(pipeline.RadianceHitgroupKernel);
                    hitgroupKernelsList.Add(pipeline.ShadowHitgroupKernel);
                }
            }
            for (var kind = 0; kind < customCounts.Length; kind++)
            {
                if (customCounts[kind] == 0)
                    continue;
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    hitgroupKernelsList.Add(pipeline.RadianceHitgroupKernelsCustom[kind]);
                    hitgroupKernelsList.Add(pipeline.ShadowHitgroupKernelsCustom[kind]);
                }
            }

            bool hasVolumeGrid = scene.VoxelMaterialIds.Length > 0;
            if (hasVolumeGrid)
            {
                // NumSbtRecords=1 for this build input (see the custom-primitives GAS
                // build) - its record's own custom data is never read (ShadeVolumeGrid
                // looks up materials directly via LaunchParams.Materials instead), so
                // only the kernel entries (for header/program-group dispatch) matter.
                hitgroupKernelsList.Add(pipeline.RadianceHitgroupKernelVolumeGrid);
                hitgroupKernelsList.Add(pipeline.ShadowHitgroupKernelVolumeGrid);
            }

            var hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernelsList);

            // Resolved once per material index - any material without a texture path
            // (the overwhelming majority of Sample13's scenes) keeps TextureObject=0,
            // matching Sample08's "0 = no texture" convention. NormalTexture/OrmTexture
            // (docs/SAMPLE15_PLAN.md Milestone M6) follow the exact same resolution as
            // BaseColorTexture - MaterialSbtData.OrmTexture/NormalTexture themselves are
            // never read here; only the parallel *TexturePaths arrays are.
            var textureHandles = new ulong[scene.Materials.Length];
            var normalTextureHandles = new ulong[scene.Materials.Length];
            var ormTextureHandles = new ulong[scene.Materials.Length];
            for (var i = 0; i < scene.Materials.Length; i++)
            {
                string path = i < scene.MaterialTexturePaths.Length ? scene.MaterialTexturePaths[i] : null;
                textureHandles[i] = string.IsNullOrEmpty(path) ? 0 : textures.GetOrLoad(path);

                string normalPath = i < scene.MaterialNormalTexturePaths.Length ? scene.MaterialNormalTexturePaths[i] : null;
                normalTextureHandles[i] = string.IsNullOrEmpty(normalPath) ? 0 : textures.GetOrLoad(normalPath);

                string ormPath = i < scene.MaterialOrmTexturePaths.Length ? scene.MaterialOrmTexturePaths[i] : null;
                ormTextureHandles[i] = string.IsNullOrEmpty(ormPath) ? 0 : textures.GetOrLoad(ormPath);
            }

            void FillMaterialRecords(int baseRecordIndex)
            {
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    var mat = scene.Materials[i];
                    ref var record = ref hitgroupRecordsArray[baseRecordIndex + (i * 2)];
                    record.BaseColor = mat.BaseColor;
                    record.Metallic = mat.Metallic;
                    record.Roughness = mat.Roughness;
                    record.IOR = mat.IOR;
                    record.Emission = mat.Emission;
                    record.EmissionStrength = mat.EmissionStrength;
                    record.Transmission = mat.Transmission;
                    record.TransmissionColor = mat.TransmissionColor;
                    record.TransmissionRoughness = mat.TransmissionRoughness;
                    record.BaseColorTexture = textureHandles[i];
                    record.OrmTexture = ormTextureHandles[i];
                    record.NormalTexture = normalTextureHandles[i];
                    record.NormalStrength = mat.NormalStrength;
                    record.TextureWeight = mat.TextureWeight;
                    record.UVScale = mat.UVScale;
                    record.MaterialKind = mat.MaterialKind;
                    record.CheckerColorB = mat.CheckerColorB;
                    record.CheckerScale = mat.CheckerScale;
                    record.AlphaCutoff = mat.AlphaCutoff;
                    record.EmissiveLightListBase = mat.EmissiveLightListBase;
                }
            }

            var recordIndex = 0;
            for (var m = 0; m < triangleMeshCount; m++)
            {
                FillMaterialRecords(recordIndex);
                recordIndex += scene.Materials.Length * 2;
            }
            for (var kind = 0; kind < customCounts.Length; kind++)
            {
                if (customCounts[kind] == 0)
                    continue;
                FillMaterialRecords(recordIndex);
                recordIndex += scene.Materials.Length * 2;
            }

            hitgroupRecordsBuffer = accelerator.Allocate1D(hitgroupRecordsArray);

            return new OptixShaderBindingTable()
            {
                RaygenRecord = pipeline.RaygenRecordsBuffer.NativePtr,
                MissRecordBase = pipeline.MissRecordsBuffer.NativePtr,
                MissRecordStrideInBytes = (uint)Marshal.SizeOf<MissRecord>(),
                MissRecordCount = (uint)pipeline.MissRecordsBuffer.Length,
                HitgroupRecordBase = hitgroupRecordsBuffer.NativePtr,
                HitgroupRecordStrideInBytes = (uint)Marshal.SizeOf<HitgroupRecord>(),
                HitgroupRecordCount = (uint)hitgroupRecordsBuffer.Length
            };
        }

        public void DisposeBuffers()
        {
            hitgroupRecordsBuffer?.Dispose();
            hitgroupRecordsBuffer = null;
        }
    }
}
