using ILGPU;
using ILGPU.OptiX;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;

namespace Sample15
{
    /// <summary>
    /// Owns the per-scene device buffers: triangle geometry, lights, and the material
    /// palette. Reallocated wholesale on every scene switch (Upload).
    /// </summary>
    public sealed class SceneGpuBuffers : IDisposable
    {
        readonly CudaAccelerator accelerator;

        public SceneGpuBuffers(CudaAccelerator accelerator)
        {
            this.accelerator = accelerator;
        }

        public MemoryBuffer1D<Vec3, Stride1D.Dense> Vertices { get; private set; }
        public MemoryBuffer1D<Vec3, Stride1D.Dense> Normals { get; private set; }
        public MemoryBuffer1D<Vec2, Stride1D.Dense> TexCoords { get; private set; }
        public MemoryBuffer1D<Vec3, Stride1D.Dense> Tangents { get; private set; }
        public MemoryBuffer1D<Vec3i, Stride1D.Dense> Indices { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> TriangleMaterialIds { get; private set; }
        public MemoryBuffer1D<PointLightGpu, Stride1D.Dense> Lights { get; private set; }

        // Unified NEE light list - see
        // Scenes/LightList.cs/LaunchParams.NeeLights.
        public MemoryBuffer1D<LightGpu, Stride1D.Dense> NeeLights { get; private set; }
        public MemoryBuffer1D<float, Stride1D.Dense> NeeLightCdf { get; private set; }
        public MemoryBuffer1D<float, Stride1D.Dense> NeeLightAreaPdf { get; private set; }

        // Allocates every buffer from the scene's host arrays. The caller must have
        // called DisposeAll() first (scene switches always tear down before uploading).
        public void Upload(SceneData scene)
        {
            // Defensive fallback -
            // Rays/RadianceRay.cs reads launchParams.Tangents[tri.x] unconditionally
            // for every triangle hit, with no NumNeeLights-style empty-buffer guard
            // (tangents are structural per-vertex mesh data, not an optional feature
            // list). A scene that constructs SceneData directly instead of going
            // through SceneBuilder (as MeshScenes.cs's BuildMeshScene did, causing a
            // real illegal-memory-access crash the first time this milestone's code
            // actually ran against it) would otherwise upload a null Tangents pointer
            // against a populated Vertices buffer. Any per-hit orthogonalization
            // quality loss from this fallback is invisible anyway, since it's only
            // reached by scenes with no NormalTexture to begin with.
            if (scene.Vertices.Length > 0 && scene.Tangents.Length != scene.Vertices.Length)
            {
                Console.WriteLine($"[SceneGpuBuffers] WARNING: scene '{scene.Name}' has {scene.Vertices.Length} vertices but {scene.Tangents.Length} tangents - synthesizing a fallback Tangents buffer. This scene should set SceneData.Tangents.");
                var fallbackTangents = new Vec3[scene.Vertices.Length];
                for (int i = 0; i < fallbackTangents.Length; i++)
                    fallbackTangents[i] = new Vec3(1f, 0f, 0f);
                scene.Tangents = fallbackTangents;
            }

            Vertices = AllocateOrNull(scene.Vertices);
            Normals = AllocateOrNull(scene.Normals);
            TexCoords = AllocateOrNull(scene.TexCoords);
            Tangents = AllocateOrNull(scene.Tangents);
            Indices = AllocateOrNull(scene.Indices);
            TriangleMaterialIds = AllocateOrNull(scene.TriangleMaterialIds);
            Lights = AllocateOrNull(scene.Lights);
            NeeLights = AllocateOrNull(scene.NeeLights);
            NeeLightCdf = AllocateOrNull(scene.NeeLightCdf);
            NeeLightAreaPdf = AllocateOrNull(scene.NeeLightAreaPdf);
        }

        // Per-frame update for orbiting/pulsing light animation.
        public void UpdateLights(PointLightGpu[] animatedLights) =>
            Lights.CopyFromCPU(animatedLights);

        public void DisposeAll()
        {
            NeeLightAreaPdf?.Dispose(); NeeLightAreaPdf = null;
            NeeLightCdf?.Dispose(); NeeLightCdf = null;
            NeeLights?.Dispose(); NeeLights = null;
            Lights?.Dispose(); Lights = null;
            TriangleMaterialIds?.Dispose(); TriangleMaterialIds = null;
            Indices?.Dispose(); Indices = null;
            Tangents?.Dispose(); Tangents = null;
            TexCoords?.Dispose(); TexCoords = null;
            Normals?.Dispose(); Normals = null;
            Vertices?.Dispose(); Vertices = null;
        }

        public void Dispose() => DisposeAll();

        // A zero-length source array must NOT be passed to Allocate1D - ILGPU's
        // MemoryBuffer1D constructor throws a NullReferenceException for a zero-element
        // allocation (its underlying device pointer comes back null and the view
        // wrapper doesn't handle that), so a scene lacking a given buffer must leave it
        // null instead of allocating.
        MemoryBuffer1D<T, Stride1D.Dense> AllocateOrNull<T>(T[] data) where T : unmanaged =>
            data.Length > 0 ? accelerator.Allocate1D(data) : null;
    }
}
