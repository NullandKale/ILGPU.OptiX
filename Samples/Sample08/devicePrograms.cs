using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Cuda;
using System.Numerics;

namespace Sample08
{
    public static class devicePrograms
    {
        public static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var camera = launchParams.camera;

            float screenX = (2f * ((ix + 0.5f) * camera.reciprocalWidth)) - 1f;
            float screenY = (2f * ((iy + 0.5f) * camera.reciprocalHeight)) - 1f;

            Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

            uint p0 = 0, p1 = 0, p2 = 0;

            OptixTrace.Trace(
                launchParams.traversable,
                (camera.origin.x, camera.origin.y, camera.origin.z),
                (rayDir.x, rayDir.y, rayDir.z),
                1e-3f,
                1e20f,
                0.0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref p0,
                ref p1,
                ref p2);

            Vec3 pixelColor = new Vec3(
                Interop.IntAsFloat(p0),
                Interop.IntAsFloat(p1),
                Interop.IntAsFloat(p2));

            uint r = (uint)(XMath.Clamp(pixelColor.x, 0f, 1f) * 255f);
            uint g = (uint)(XMath.Clamp(pixelColor.y, 0f, 1f) * 255f);
            uint b = (uint)(XMath.Clamp(pixelColor.z, 0f, 1f) * 255f);

            uint rgba = 0xff000000 | (b << 0) | (g << 8) | (r << 16);

            long fbIndex = ix + iy * launchParams.camera.width;
            launchParams.ColorBuffer[fbIndex] = rgba;
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            SetPRD(launchParams.camera.noHitColor);
        }

        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            // One merged GAS build input for the whole model geometry (see Model.cs),
            // but per-material data comes from the current hitgroup record's custom
            // data via OptixGetSbtDataPointer. This barycentric-interpolates the
            // shading normal and texcoord, then - if the hit material has a texture -
            // modulates the material's base color by a bindless CUDA texture-object
            // sample (see CudaTextureObject.cs/CudaTex2D.cs).
            uint primId = OptixGetPrimitiveIndex.Value;
            Vec3i tri = launchParams.Indices[primId];
            var (bw, bu, bv) = OptixHitProgramHelpers.GetTriangleBarycentrics();

            Vec3 n0 = launchParams.Normals[tri.x];
            Vec3 n1 = launchParams.Normals[tri.y];
            Vec3 n2 = launchParams.Normals[tri.z];
            var normalV3 = OptixHitProgramHelpers.InterpolateAttribute(
                new Vector3(n0.x, n0.y, n0.z),
                new Vector3(n1.x, n1.y, n1.z),
                new Vector3(n2.x, n2.y, n2.z),
                bw, bu, bv);
            Vec3 normal = new Vec3(
                Vector3.Normalize(normalV3).X,
                Vector3.Normalize(normalV3).Y,
                Vector3.Normalize(normalV3).Z);

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            Vec3 diffuseColor = sbtData->Color;
            ulong textureObject = sbtData->TextureObject;

            if (textureObject != 0)
            {
                Vec2 t0 = launchParams.TexCoords[tri.x];
                Vec2 t1 = launchParams.TexCoords[tri.y];
                Vec2 t2 = launchParams.TexCoords[tri.z];
                float tu = (bw * t0.x) + (bu * t1.x) + (bv * t2.x);
                float tv = (bw * t0.y) + (bu * t1.y) + (bv * t2.y);

                var (tr, tg, tb, _) = CudaTex2D.Sample(textureObject, tu, tv);
                diffuseColor = new Vec3(diffuseColor.x * tr, diffuseColor.y * tg, diffuseColor.z * tb);
            }

            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            float cosDN = 0.2f + (0.8f * XMath.Abs(Vec3.dot(rayDir, normal)));
            SetPRD(cosDN * diffuseColor);
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        private static void SetPRD(Vec3 color)
        {
            OptixPayloadInterop.SetFloat(0, color.x);
            OptixPayloadInterop.SetFloat(1, color.y);
            OptixPayloadInterop.SetFloat(2, color.z);
        }

        public static void flipBitmap(Index1D index, int width, int height, ArrayView<byte> source, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);

            int newIndex = ((y * width) + x) * 4;
            int oldIndexStart = index * 4;

            dest[newIndex] = source[oldIndexStart];
            dest[newIndex + 1] = source[oldIndexStart + 1];
            dest[newIndex + 2] = source[oldIndexStart + 2];
            dest[newIndex + 3] = source[oldIndexStart + 3];
        }
    }
}
