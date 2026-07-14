using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;

namespace Sample14
{
    // One __intersection__ program per custom-primitive type - kept in its own file
    // since more primitive types (Box, CylinderY, Disk, rects) accumulate here in M4.
    public static class IntersectionPrograms
    {
        // Custom-primitive hit-kind tags passed to OptixReportIntersection - must stay
        // below OptixGetHitKind.TriangleFrontFace (0xFE), the lowest reserved built-in
        // triangle-hit value. __closest__radiance uses this to tell primitive kinds
        // apart without a separate "which build input was hit" query.
        public const uint HitKindSphere = 0;
        public const uint HitKindBox = 1;
        public const uint HitKindCylinderY = 2;
        public const uint HitKindDisk = 3;
        public const uint HitKindXYRect = 4;
        public const uint HitKindXZRect = 5;
        public const uint HitKindYZRect = 6;

        // The volume grid is a single GAS primitive (one AABB spanning the whole grid),
        // so its hitKind must also carry which face the DDA loop's last step crossed
        // (0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z - see devicePrograms.cs's
        // VolumeGridFaceNormal), unlike every other primitive kind above where hitKind
        // alone identifies the kind. Occupies hitKind values [7,12].
        public const uint HitKindVolumeGridFaceBase = 7;

        public unsafe static void __intersection__sphere(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            SphereData sphere = launchParams.Spheres[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayOrigin = new Vec3(ox, oy, oz);
            Vec3 rayDir = new Vec3(dx, dy, dz);

            Vec3 oc = rayOrigin - sphere.Center;
            float a = Vec3.dot(rayDir, rayDir);
            float halfB = Vec3.dot(oc, rayDir);
            float c = oc.lengthSquared() - (sphere.Radius * sphere.Radius);
            float discriminant = (halfB * halfB) - (a * c);
            if (discriminant < 0f)
                return;

            // Report both algebraic roots (near then far) - OptiX itself only accepts
            // whichever call(s) fall within the ray's current valid [tmin,tmax] and
            // keeps the closest one across possibly-multiple accepted reports from this
            // one intersection-program invocation, so no manual tmin/tmax bookkeeping is
            // needed here (matches standard OptiX custom-sphere-intersection practice).
            float sqrtD = XMath.Sqrt(discriminant);
            float invA = 1f / a;
            float nearRoot = (-halfB - sqrtD) * invA;
            float farRoot = (-halfB + sqrtD) * invA;

            OptixReportIntersection.Report(nearRoot, HitKindSphere);
            OptixReportIntersection.Report(farRoot, HitKindSphere);
        }

        // 3-axis slab test (direct port of the standard ray-AABB intersection, used in
        // place of the reference's 6-rect-face proxy Box).
        public unsafe static void __intersection__box(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            BoxData box = launchParams.Boxes[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;

            float tNear = float.NegativeInfinity;
            float tFar = float.PositiveInfinity;

            if (!SlabAxis(ox, dx, box.Min.x, box.Max.x, ref tNear, ref tFar))
                return;
            if (!SlabAxis(oy, dy, box.Min.y, box.Max.y, ref tNear, ref tFar))
                return;
            if (!SlabAxis(oz, dz, box.Min.z, box.Max.z, ref tNear, ref tFar))
                return;

            OptixReportIntersection.Report(tNear, HitKindBox);
            OptixReportIntersection.Report(tFar, HitKindBox);
        }

        private static bool SlabAxis(float origin, float dir, float min, float max, ref float tNear, ref float tFar)
        {
            if (XMath.Abs(dir) < 1e-12f)
                return origin >= min && origin <= max;

            float invD = 1f / dir;
            float t0 = (min - origin) * invD;
            float t1 = (max - origin) * invD;
            if (t0 > t1)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tNear = XMath.Max(tNear, t0);
            tFar = XMath.Min(tFar, t1);
            return tFar > tNear;
        }

        // Infinite cylinder along Y (quadratic in X/Z only) plus two cap disks (direct
        // port of RayTracing/Objects/BoundedObjects.cs's CylinderY).
        public unsafe static void __intersection__cylinderY(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            CylinderYData cyl = launchParams.CylindersY[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayOrigin = new Vec3(ox, oy, oz);
            Vec3 rayDir = new Vec3(dx, dy, dz);

            float ocx = ox - cyl.Center.x;
            float ocz = oz - cyl.Center.z;
            float a = (dx * dx) + (dz * dz);
            if (a > 1e-12f)
            {
                float halfB = (ocx * dx) + (ocz * dz);
                float c = (ocx * ocx) + (ocz * ocz) - (cyl.Radius * cyl.Radius);
                float discriminant = (halfB * halfB) - (a * c);
                if (discriminant >= 0f)
                {
                    float sqrtD = XMath.Sqrt(discriminant);
                    float invA = 1f / a;
                    float t0 = (-halfB - sqrtD) * invA;
                    float t1 = (-halfB + sqrtD) * invA;

                    float y0 = oy + (t0 * dy);
                    if (y0 >= cyl.YMin && y0 <= cyl.YMax)
                        OptixReportIntersection.Report(t0, HitKindCylinderY);

                    float y1 = oy + (t1 * dy);
                    if (y1 >= cyl.YMin && y1 <= cyl.YMax)
                        OptixReportIntersection.Report(t1, HitKindCylinderY);
                }
            }

            if (cyl.Capped != 0 && XMath.Abs(dy) > 1e-12f)
            {
                float tTop = (cyl.YMax - oy) / dy;
                Vec3 pTop = rayOrigin + (tTop * rayDir);
                float rTop2 = ((pTop.x - cyl.Center.x) * (pTop.x - cyl.Center.x)) + ((pTop.z - cyl.Center.z) * (pTop.z - cyl.Center.z));
                if (rTop2 <= cyl.Radius * cyl.Radius)
                    OptixReportIntersection.Report(tTop, HitKindCylinderY);

                float tBottom = (cyl.YMin - oy) / dy;
                Vec3 pBottom = rayOrigin + (tBottom * rayDir);
                float rBottom2 = ((pBottom.x - cyl.Center.x) * (pBottom.x - cyl.Center.x)) + ((pBottom.z - cyl.Center.z) * (pBottom.z - cyl.Center.z));
                if (rBottom2 <= cyl.Radius * cyl.Radius)
                    OptixReportIntersection.Report(tBottom, HitKindCylinderY);
            }
        }

        // Plane-divide + radial cutoff (direct port of RayTracing/Objects/Surfaces.cs's
        // Disk).
        public unsafe static void __intersection__disk(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            DiskData disk = launchParams.Disks[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayOrigin = new Vec3(ox, oy, oz);
            Vec3 rayDir = new Vec3(dx, dy, dz);

            float denom = Vec3.dot(disk.Normal, rayDir);
            if (XMath.Abs(denom) < 1e-12f)
                return;

            float t = Vec3.dot(disk.Center - rayOrigin, disk.Normal) / denom;
            Vec3 p = rayOrigin + (t * rayDir);
            if ((p - disk.Center).lengthSquared() <= disk.Radius * disk.Radius)
                OptixReportIntersection.Report(t, HitKindDisk);
        }

        // XYRect: fixed Z=C, spans X in [A0,A1], Y in [B0,B1] (direct port of
        // RayTracing/Objects/Surfaces.cs's XYRect).
        public unsafe static void __intersection__xyRect(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            RectData rect = launchParams.XYRects[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            if (XMath.Abs(dz) < 1e-12f)
                return;

            float t = (rect.C - oz) / dz;
            float x = ox + (t * dx);
            float y = oy + (t * dy);
            if (x >= rect.A0 && x <= rect.A1 && y >= rect.B0 && y <= rect.B1)
                OptixReportIntersection.Report(t, HitKindXYRect);
        }

        // XZRect: fixed Y=C, spans X in [A0,A1], Z in [B0,B1].
        public unsafe static void __intersection__xzRect(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            RectData rect = launchParams.XZRects[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            if (XMath.Abs(dy) < 1e-12f)
                return;

            float t = (rect.C - oy) / dy;
            float x = ox + (t * dx);
            float z = oz + (t * dz);
            if (x >= rect.A0 && x <= rect.A1 && z >= rect.B0 && z <= rect.B1)
                OptixReportIntersection.Report(t, HitKindXZRect);
        }

        // YZRect: fixed X=C, spans Y in [A0,A1], Z in [B0,B1].
        public unsafe static void __intersection__yzRect(LaunchParams launchParams)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            RectData rect = launchParams.YZRects[primId];

            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            if (XMath.Abs(dx) < 1e-12f)
                return;

            float t = (rect.C - ox) / dx;
            float y = oy + (t * dy);
            float z = oz + (t * dz);
            if (y >= rect.A0 && y <= rect.A1 && z >= rect.B0 && z <= rect.B1)
                OptixReportIntersection.Report(t, HitKindYZRect);
        }

        // Amanatides & Woo fast voxel traversal against a flat row-major solid-voxel
        // occupancy grid (the DDA *technique* is ported from the reference's
        // VolumeGrid.cs; its 8x8x8 Morton-brick storage is a CPU-cache optimization not
        // carried over here, a flat array suits GPU access patterns better). The whole
        // grid is ONE custom
        // primitive (NumPrimitives=1 in its GAS build input) - the loop walks voxels
        // entirely inside this one intersection-program invocation.
        public unsafe static void __intersection__volumeGrid(LaunchParams launchParams)
        {
            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;

            Vec3 gridMin = launchParams.VolumeGridMin;
            Vec3 voxelSize = launchParams.VolumeVoxelSize;
            Vec3i dims = launchParams.VolumeDims;
            float gridMaxX = gridMin.x + (dims.x * voxelSize.x);
            float gridMaxY = gridMin.y + (dims.y * voxelSize.y);
            float gridMaxZ = gridMin.z + (dims.z * voxelSize.z);

            float tEnter = float.NegativeInfinity;
            float tExit = float.PositiveInfinity;
            if (!SlabAxis(ox, dx, gridMin.x, gridMaxX, ref tEnter, ref tExit))
                return;
            if (!SlabAxis(oy, dy, gridMin.y, gridMaxY, ref tEnter, ref tExit))
                return;
            if (!SlabAxis(oz, dz, gridMin.z, gridMaxZ, ref tEnter, ref tExit))
                return;
            if (tExit <= 0f)
                return;

            float tStart = XMath.Max(tEnter, 0f);
            float px = ox + (tStart * dx);
            float py = oy + (tStart * dy);
            float pz = oz + (tStart * dz);

            int ix = XMath.Clamp((int)XMath.Floor((px - gridMin.x) / voxelSize.x), 0, dims.x - 1);
            int iy = XMath.Clamp((int)XMath.Floor((py - gridMin.y) / voxelSize.y), 0, dims.y - 1);
            int iz = XMath.Clamp((int)XMath.Floor((pz - gridMin.z) / voxelSize.z), 0, dims.z - 1);

            int stepX = dx > 0f ? 1 : -1;
            int stepY = dy > 0f ? 1 : -1;
            int stepZ = dz > 0f ? 1 : -1;

            float tMaxX = VoxelBoundaryT(ox, dx, gridMin.x, voxelSize.x, ix, stepX);
            float tMaxY = VoxelBoundaryT(oy, dy, gridMin.y, voxelSize.y, iy, stepY);
            float tMaxZ = VoxelBoundaryT(oz, dz, gridMin.z, voxelSize.z, iz, stepZ);

            float tDeltaX = XMath.Abs(dx) > 1e-12f ? voxelSize.x / XMath.Abs(dx) : float.PositiveInfinity;
            float tDeltaY = XMath.Abs(dy) > 1e-12f ? voxelSize.y / XMath.Abs(dy) : float.PositiveInfinity;
            float tDeltaZ = XMath.Abs(dz) > 1e-12f ? voxelSize.z / XMath.Abs(dz) : float.PositiveInfinity;

            // Default face (used only if the very first voxel checked is already
            // occupied, i.e. the ray origin starts inside solid geometry - not expected
            // in a well-formed scene, so an arbitrary but harmless +Y fallback is fine).
            uint faceCode = 2;
            float t = tStart;

            int maxSteps = dims.x + dims.y + dims.z + 1;
            for (int step = 0; step < maxSteps; step++)
            {
                uint voxel = launchParams.VoxelMaterialIds[(ix * dims.y * dims.z) + (iy * dims.z) + iz];
                if (voxel != 0)
                {
                    OptixReportIntersection.Report(t, HitKindVolumeGridFaceBase + faceCode);
                    return;
                }

                if (tMaxX < tMaxY && tMaxX < tMaxZ)
                {
                    ix += stepX;
                    if (ix < 0 || ix >= dims.x)
                        return;
                    t = tMaxX;
                    tMaxX += tDeltaX;
                    faceCode = stepX > 0 ? 1u : 0u;
                }
                else if (tMaxY < tMaxZ)
                {
                    iy += stepY;
                    if (iy < 0 || iy >= dims.y)
                        return;
                    t = tMaxY;
                    tMaxY += tDeltaY;
                    faceCode = stepY > 0 ? 3u : 2u;
                }
                else
                {
                    iz += stepZ;
                    if (iz < 0 || iz >= dims.z)
                        return;
                    t = tMaxZ;
                    tMaxZ += tDeltaZ;
                    faceCode = stepZ > 0 ? 5u : 4u;
                }
            }
        }

        private static float VoxelBoundaryT(float origin, float dir, float gridMinAxis, float voxelSizeAxis, int voxelIndex, int step)
        {
            if (XMath.Abs(dir) < 1e-12f)
                return float.PositiveInfinity;
            float boundary = gridMinAxis + ((voxelIndex + (step > 0 ? 1 : 0)) * voxelSizeAxis);
            return (boundary - origin) / dir;
        }
    }
}
