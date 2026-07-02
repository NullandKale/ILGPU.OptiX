using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;

namespace Sample13
{
    /// <summary>
    /// Owns the per-scene device buffers: triangle geometry, per-kind custom-primitive
    /// data + AABBs + SBT-index buffers, lights, the volume grid, and the material
    /// palette. Reallocated wholesale on every scene switch (Upload), with a targeted
    /// per-frame update path for animated spheres.
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
        public MemoryBuffer1D<Vec3i, Stride1D.Dense> Indices { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> TriangleMaterialIds { get; private set; }
        public MemoryBuffer1D<PointLightGpu, Stride1D.Dense> Lights { get; private set; }

        public MemoryBuffer1D<SphereData, Stride1D.Dense> Spheres { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> SphereMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> SphereAabbs { get; private set; }

        public MemoryBuffer1D<BoxData, Stride1D.Dense> Boxes { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> BoxMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> BoxAabbs { get; private set; }

        public MemoryBuffer1D<CylinderYData, Stride1D.Dense> CylindersY { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> CylinderYMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> CylinderYAabbs { get; private set; }

        public MemoryBuffer1D<DiskData, Stride1D.Dense> Disks { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> DiskMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> DiskAabbs { get; private set; }

        public MemoryBuffer1D<RectData, Stride1D.Dense> XYRects { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> XYRectMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> XYRectAabbs { get; private set; }

        public MemoryBuffer1D<RectData, Stride1D.Dense> XZRects { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> XZRectMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> XZRectAabbs { get; private set; }

        public MemoryBuffer1D<RectData, Stride1D.Dense> YZRects { get; private set; }
        public MemoryBuffer1D<uint, Stride1D.Dense> YZRectMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> YZRectAabbs { get; private set; }

        public MemoryBuffer1D<uint, Stride1D.Dense> VoxelMaterialIds { get; private set; }
        public MemoryBuffer1D<OptixAabb, Stride1D.Dense> VolumeGridAabb { get; private set; }

        // Device-side copy of the active scene's Materials[] palette - see
        // LaunchParams.Materials.
        public MemoryBuffer1D<MaterialSbtData, Stride1D.Dense> Materials { get; private set; }

        // Allocates every buffer from the scene's host arrays. The caller must have
        // called DisposeAll() first (scene switches always tear down before uploading).
        public void Upload(SceneData scene)
        {
            Vertices = AllocateOrNull(scene.Vertices);
            Normals = AllocateOrNull(scene.Normals);
            TexCoords = AllocateOrNull(scene.TexCoords);
            Indices = AllocateOrNull(scene.Indices);
            TriangleMaterialIds = AllocateOrNull(scene.TriangleMaterialIds);
            Lights = AllocateOrNull(scene.Lights);

            Spheres = AllocateOrNull(scene.Spheres);
            SphereMaterialIds = AllocateOrNull(scene.SphereMaterialIds);
            SphereAabbs = AllocateOrNull(ComputeSphereAabbs(scene.Spheres));

            Boxes = AllocateOrNull(scene.Boxes);
            BoxMaterialIds = AllocateOrNull(scene.BoxMaterialIds);
            BoxAabbs = AllocateOrNull(ComputeBoxAabbs(scene.Boxes));

            CylindersY = AllocateOrNull(scene.CylindersY);
            CylinderYMaterialIds = AllocateOrNull(scene.CylinderYMaterialIds);
            CylinderYAabbs = AllocateOrNull(ComputeCylinderYAabbs(scene.CylindersY));

            Disks = AllocateOrNull(scene.Disks);
            DiskMaterialIds = AllocateOrNull(scene.DiskMaterialIds);
            DiskAabbs = AllocateOrNull(ComputeDiskAabbs(scene.Disks));

            XYRects = AllocateOrNull(scene.XYRects);
            XYRectMaterialIds = AllocateOrNull(scene.XYRectMaterialIds);
            XYRectAabbs = AllocateOrNull(ComputeXYRectAabbs(scene.XYRects));

            XZRects = AllocateOrNull(scene.XZRects);
            XZRectMaterialIds = AllocateOrNull(scene.XZRectMaterialIds);
            XZRectAabbs = AllocateOrNull(ComputeXZRectAabbs(scene.XZRects));

            YZRects = AllocateOrNull(scene.YZRects);
            YZRectMaterialIds = AllocateOrNull(scene.YZRectMaterialIds);
            YZRectAabbs = AllocateOrNull(ComputeYZRectAabbs(scene.YZRects));

            VoxelMaterialIds = AllocateOrNull(scene.VoxelMaterialIds);
            VolumeGridAabb = AllocateOrNull(ComputeVolumeGridAabb(scene));
            Materials = AllocateOrNull(scene.Materials);
        }

        // Per-frame update for bobbing-sphere animation: refreshes the sphere data and
        // AABB buffer CONTENTS in place (same device addresses, required by the GAS
        // refit - see AccelStructureBuilder.RefitCustomPrimitives).
        public void UpdateAnimatedSpheres(SphereData[] animatedSpheres)
        {
            Spheres.CopyFromCPU(animatedSpheres);
            SphereAabbs.CopyFromCPU(ComputeSphereAabbs(animatedSpheres));
        }

        // Per-frame update for orbiting/pulsing light animation.
        public void UpdateLights(PointLightGpu[] animatedLights) =>
            Lights.CopyFromCPU(animatedLights);

        public void DisposeAll()
        {
            Materials?.Dispose(); Materials = null;
            VolumeGridAabb?.Dispose(); VolumeGridAabb = null;
            VoxelMaterialIds?.Dispose(); VoxelMaterialIds = null;
            YZRectAabbs?.Dispose(); YZRectAabbs = null;
            YZRectMaterialIds?.Dispose(); YZRectMaterialIds = null;
            YZRects?.Dispose(); YZRects = null;
            XZRectAabbs?.Dispose(); XZRectAabbs = null;
            XZRectMaterialIds?.Dispose(); XZRectMaterialIds = null;
            XZRects?.Dispose(); XZRects = null;
            XYRectAabbs?.Dispose(); XYRectAabbs = null;
            XYRectMaterialIds?.Dispose(); XYRectMaterialIds = null;
            XYRects?.Dispose(); XYRects = null;
            DiskAabbs?.Dispose(); DiskAabbs = null;
            DiskMaterialIds?.Dispose(); DiskMaterialIds = null;
            Disks?.Dispose(); Disks = null;
            CylinderYAabbs?.Dispose(); CylinderYAabbs = null;
            CylinderYMaterialIds?.Dispose(); CylinderYMaterialIds = null;
            CylindersY?.Dispose(); CylindersY = null;
            BoxAabbs?.Dispose(); BoxAabbs = null;
            BoxMaterialIds?.Dispose(); BoxMaterialIds = null;
            Boxes?.Dispose(); Boxes = null;
            SphereAabbs?.Dispose(); SphereAabbs = null;
            SphereMaterialIds?.Dispose(); SphereMaterialIds = null;
            Spheres?.Dispose(); Spheres = null;

            Lights?.Dispose(); Lights = null;
            TriangleMaterialIds?.Dispose(); TriangleMaterialIds = null;
            Indices?.Dispose(); Indices = null;
            TexCoords?.Dispose(); TexCoords = null;
            Normals?.Dispose(); Normals = null;
            Vertices?.Dispose(); Vertices = null;
        }

        public void Dispose() => DisposeAll();

        // A zero-length source array must NOT be passed to Allocate1D - ILGPU's
        // MemoryBuffer1D constructor throws a NullReferenceException for a zero-element
        // allocation (its underlying device pointer comes back null and the view
        // wrapper doesn't handle that), so a scene lacking a given primitive kind (e.g.
        // the volume-grid test scene has zero triangles, and the debug scene has no
        // volume grid) must leave that buffer null instead of allocating.
        MemoryBuffer1D<T, Stride1D.Dense> AllocateOrNull<T>(T[] data) where T : unmanaged =>
            data.Length > 0 ? accelerator.Allocate1D(data) : null;

        public static IntPtr NativePtrOrZero<T>(MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged =>
            buffer?.NativePtr ?? IntPtr.Zero;

        static OptixAabb[] ComputeSphereAabbs(SphereData[] spheres)
        {
            var result = new OptixAabb[spheres.Length];
            for (var i = 0; i < spheres.Length; i++)
            {
                var s = spheres[i];
                result[i] = new OptixAabb
                {
                    MinX = s.Center.x - s.Radius, MinY = s.Center.y - s.Radius, MinZ = s.Center.z - s.Radius,
                    MaxX = s.Center.x + s.Radius, MaxY = s.Center.y + s.Radius, MaxZ = s.Center.z + s.Radius,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeBoxAabbs(BoxData[] boxes)
        {
            var result = new OptixAabb[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            {
                var b = boxes[i];
                result[i] = new OptixAabb
                {
                    MinX = b.Min.x, MinY = b.Min.y, MinZ = b.Min.z,
                    MaxX = b.Max.x, MaxY = b.Max.y, MaxZ = b.Max.z,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeCylinderYAabbs(CylinderYData[] cylinders)
        {
            var result = new OptixAabb[cylinders.Length];
            for (var i = 0; i < cylinders.Length; i++)
            {
                var c = cylinders[i];
                result[i] = new OptixAabb
                {
                    MinX = c.Center.x - c.Radius, MinY = c.YMin, MinZ = c.Center.z - c.Radius,
                    MaxX = c.Center.x + c.Radius, MaxY = c.YMax, MaxZ = c.Center.z + c.Radius,
                };
            }
            return result;
        }

        // A disk's AABB depends on its (arbitrary) orientation - a spherical bound
        // around its center is a simple, always-correct (if occasionally loose) choice.
        static OptixAabb[] ComputeDiskAabbs(DiskData[] disks)
        {
            var result = new OptixAabb[disks.Length];
            for (var i = 0; i < disks.Length; i++)
            {
                var d = disks[i];
                result[i] = new OptixAabb
                {
                    MinX = d.Center.x - d.Radius, MinY = d.Center.y - d.Radius, MinZ = d.Center.z - d.Radius,
                    MaxX = d.Center.x + d.Radius, MaxY = d.Center.y + d.Radius, MaxZ = d.Center.z + d.Radius,
                };
            }
            return result;
        }

        const float RectAabbEpsilon = 1e-4f; // finite thickness along the fixed axis

        static OptixAabb[] ComputeXYRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.A0, MinY = r.B0, MinZ = r.C - RectAabbEpsilon,
                    MaxX = r.A1, MaxY = r.B1, MaxZ = r.C + RectAabbEpsilon,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeXZRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.A0, MinY = r.C - RectAabbEpsilon, MinZ = r.B0,
                    MaxX = r.A1, MaxY = r.C + RectAabbEpsilon, MaxZ = r.B1,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeYZRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.C - RectAabbEpsilon, MinY = r.A0, MinZ = r.B0,
                    MaxX = r.C + RectAabbEpsilon, MaxY = r.A1, MaxZ = r.B1,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeVolumeGridAabb(SceneData scene)
        {
            if (scene.VoxelMaterialIds.Length == 0)
                return Array.Empty<OptixAabb>();

            var min = scene.VolumeGridMin;
            var size = scene.VolumeVoxelSize;
            var dims = scene.VolumeDims;
            return new[]
            {
                new OptixAabb
                {
                    MinX = min.x, MinY = min.y, MinZ = min.z,
                    MaxX = min.x + (dims.x * size.x), MaxY = min.y + (dims.y * size.y), MaxZ = min.z + (dims.z * size.z),
                },
            };
        }
    }
}
