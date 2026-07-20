// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: RenderWindow.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Native;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Sample17
{
    /// <summary>
    /// Transforms &amp; hit-introspection API surface sample. Covers:
    ///  - rigid-body motion blur via a matrix motion transform
    ///    (OptixMatrixMotionTransform / OptixMotionTransformBuilder) wrapping a
    ///    static GAS - left tile, live;
    ///  - static instance-transform composition (object&lt;-&gt;world point/normal via
    ///    OptixTransforms), on-device vertex fetch (OptixGetTriangleVertexData,
    ///    both current-hit and FromHandle), instance introspection, and a
    ///    HitObject completion round-trip (Traverse -&gt; capture traverse data -&gt;
    ///    MakeNop -&gt; Make -&gt; Invoke) - middle tile, live;
    ///  - Shader Execution Reordering (optixTraverse + optixReorder + optixInvoke as
    ///    a 3-step alternative to a single optixTrace, compared per-pixel against the
    ///    classic-trace result every frame) - right tile, live.
    ///
    /// Left tile: the mesh spins ~90 degrees and translates while OptixTrace's
    /// rayTime argument sweeps back and forth between the transform's two motion
    /// keys, driven by real time. Space toggles the animation on/off (frozen at the
    /// first motion key) as a bonus contrast control - motion is on by default.
    ///
    /// Middle tile: two instances of one triangle under different
    /// scale+rotate+translate transforms, shaded on-device from the fetched vertex
    /// data and the transform composition (world normal via inverse-transpose),
    /// tinted per optixGetInstanceId. Startup runs a 1x1 check launch comparing
    /// every device result against host-computed expectations (including the
    /// HitObject traverse-data capture / MakeNop / Make reconstruction round-trip)
    /// and prints PASS/FAIL per check.
    ///
    /// Right tile: every pixel renders the same hit twice - once via classic
    /// OptixTrace.Trace, once via OptixHitObject.Traverse -&gt; OptixReorder.Invoke -&gt;
    /// OptixHitObject.Invoke - through the identical hitgroup/miss program. SER only
    /// affects scheduling/performance, not correctness, so the two must be
    /// pixel-identical; any mismatch is drawn in bright magenta so a regression
    /// would be immediately obvious.
    ///
    /// [Controls] Hold Left Mouse + drag to orbit (orbits all three tiles together).
    /// Space toggles motion (left tile only, bonus control - on by default). Esc to
    /// quit.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        // ---- Right tile: Shader Execution Reordering ----

        private struct SerLaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public OptixDeviceView<uint> MismatchBuffer;
            public int Width;
            public int Height;
            public int XOffset;
            public int Stride;
            public ulong Traversable;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        private struct SerRadiancePayload
        {
            public uint P0, P1, P2;
        }

        private static uint SerHash(uint x)
        {
            x = (x ^ 61u) ^ (x >> 16);
            x *= 9u;
            x ^= x >> 4;
            x *= 0x27d4eb2du;
            x ^= x >> 15;
            return x;
        }

        private static void SerRaygen(SerLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            float aspect = (float)launchParams.Width / launchParams.Height;
            float ndcX = (2f * ((ix + 0.5f) / launchParams.Width) - 1f) * aspect;
            float ndcY = 1f - 2f * ((iy + 0.5f) / launchParams.Height);

            float dx = launchParams.ForwardX * launchParams.Focal + launchParams.RightX * ndcX + launchParams.UpX * ndcY;
            float dy = launchParams.ForwardY * launchParams.Focal + launchParams.RightY * ndcX + launchParams.UpY * ndcY;
            float dz = launchParams.ForwardZ * launchParams.Focal + launchParams.RightZ * ndcX + launchParams.UpZ * ndcY;
            float invLen = 1f / XMath.Sqrt(dx * dx + dy * dy + dz * dz);
            dx *= invLen;
            dy *= invLen;
            dz *= invLen;

            // Path A: classic optixTrace.
            uint a0 = 0, a1 = 0, a2 = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                1e-3f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref a0,
                ref a1,
                ref a2);

            // Path B: SER - traverse, reorder, then invoke the same hitgroup/miss.
            uint b0 = 0, b1 = 0, b2 = 0;
            OptixHitObject.Traverse(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                1e-3f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref b0,
                ref b1,
                ref b2);
            OptixReorder.Invoke();
            OptixHitObject.Invoke(ref b0, ref b1, ref b2);

            bool mismatch = a0 != b0 || a1 != b1 || a2 != b2;

            uint ru, gu, bu;
            if (mismatch)
            {
                ru = 255; gu = 0; bu = 255;
            }
            else
            {
                float r = Interop.IntAsFloat(b0);
                float g = Interop.IntAsFloat(b1);
                float b = Interop.IntAsFloat(b2);
                ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
                gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
                bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);
            }

            uint rgba = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);

            long localPixel = ix + (long)iy * launchParams.Width;
            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            launchParams.ColorBuffer[pixel] = rgba;
            launchParams.MismatchBuffer[localPixel] = mismatch ? 1u : 0u;
        }

        private static void SerMiss(SerLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.08f);
            OptixPayloadInterop.SetFloat(1, 0.08f);
            OptixPayloadInterop.SetFloat(2, 0.12f);
        }

        private static void SerClosestHit(SerLaunchParams launchParams)
        {
            // Hash the hit triangle's primitive index into a distinct color per
            // triangle - a stand-in for "many divergent BSDFs" (this project's
            // shading is flat-color only).
            uint h = SerHash(OptixGetPrimitiveIndex.Value * 2654435761u);
            float r = ((h >> 0) & 0xFFu) / 255f;
            float g = ((h >> 8) & 0xFFu) / 255f;
            float b = ((h >> 16) & 0xFFu) / 255f;
            OptixPayloadInterop.SetFloat(0, r);
            OptixPayloadInterop.SetFloat(1, g);
            OptixPayloadInterop.SetFloat(2, b);
        }

        // ---- Mode 2: static transforms / vertex fetch / HitObject completion ----

        private static readonly Vector3 XformV0 = new Vector3(0f, 0f, 0f);
        private static readonly Vector3 XformV1 = new Vector3(1f, 0f, 0f);
        private static readonly Vector3 XformV2 = new Vector3(0f, 1f, 0f);

        // Instance 0: scale 2, rotate 90 degrees about +Y ((x,y,z) -> (z,y,-x)),
        // translate (2,1,-1). Row-major 3x4:
        //   world V0 = (2,1,-1), world V1 = (2,1,-3), world V2 = (2,3,-1)
        private static readonly float[] XformInstanceTransform0 =
        {
            0f, 0f, 2f, 2f,
            0f, 2f, 0f, 1f,
            -2f, 0f, 0f, -1f,
        };

        // Instance 1: scale 1.5, no rotation, translate (-1, 0.5, 1).
        private static readonly float[] XformInstanceTransform1 =
        {
            1.5f, 0f, 0f, -1f,
            0f, 1.5f, 0f, 0.5f,
            0f, 0f, 1.5f, 1f,
        };

        private const uint XformInstanceId0 = 77;
        private const uint XformInstanceId1 = 78;

        // Check ray towards +X hitting instance 0's world triangle (x=2 plane).
        private const float XformCheckOriginX = -5f, XformCheckOriginY = 1.5f, XformCheckOriginZ = -1.6f;
        private const float XformExpectedHitT = 7f;

        private const int SlotChRuns = 0;
        private const int SlotVertexDataOk = 1;
        private const int SlotVertexFromHandleOk = 2;
        private const int SlotObjectToWorldOk = 3;
        private const int SlotWorldToObjectOk = 4;
        private const int SlotNormalOk = 5;
        private const int SlotInstanceId = 6;
        private const int SlotInstanceIndex = 7;
        private const int SlotTransformListSize = 8;
        private const int SlotSerIsHit = 9;
        private const int SlotSerInstanceId = 10;
        private const int SlotSerNopOk = 11;
        private const int SlotSerRemadeIsHit = 12;
        private const int SlotSerWorldToObjectOk = 13;
        private const int SlotSerSbtRecordIndex = 14;
        private const int SlotHitT = 15;
        private const int SlotCount = 16;

        private struct XformLaunchParams
        {
            public OptixDeviceView<uint> ResultBuffer;
            public OptixDeviceView<uint> ColorBuffer;
            public OptixDeviceView<ulong> TransformHandleScratch;
            public ulong TransformHandleScratchAddress;
            public ulong Traversable;
            public int CheckMode;
            public int XOffset;
            public int Stride;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        private struct XformRadiancePayload
        {
            public uint Hit, R, G, B;
        }

        private static bool XformApprox(Vector3 a, Vector3 b) =>
            XMath.Abs(a.X - b.X) < 1e-3f &&
            XMath.Abs(a.Y - b.Y) < 1e-3f &&
            XMath.Abs(a.Z - b.Z) < 1e-3f;

        private static void XformRaygen(XformLaunchParams launchParams)
        {
            if (launchParams.CheckMode != 0)
            {
                XformRaygenChecks(launchParams);
                return;
            }

            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var (width, height, _) = OptixGetLaunchDimensions.Value;

            float aspect = (float)width / height;
            float ndcX = (2f * ((ix + 0.5f) / width) - 1f) * aspect;
            float ndcY = 1f - 2f * ((iy + 0.5f) / height);

            float dx = launchParams.ForwardX * launchParams.Focal + launchParams.RightX * ndcX + launchParams.UpX * ndcY;
            float dy = launchParams.ForwardY * launchParams.Focal + launchParams.RightY * ndcX + launchParams.UpY * ndcY;
            float dz = launchParams.ForwardZ * launchParams.Focal + launchParams.RightZ * ndcX + launchParams.UpZ * ndcY;
            float invLen = 1f / XMath.Sqrt(dx * dx + dy * dy + dz * dz);
            dx *= invLen;
            dy *= invLen;
            dz *= invLen;

            uint hit = 0, r = 0, g = 0, b = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                0.01f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref hit, ref r, ref g, ref b);

            uint bgra;
            if (hit != 0)
            {
                bgra = 0xff000000 | (b << 0) | (g << 8) | (r << 16);
            }
            else
            {
                // Background gradient.
                uint shade = (uint)(35f + 60f * iy / height);
                bgra = 0xff000000 | (shade + 25) | (shade << 8) | ((shade - 15) << 16);
            }

            launchParams.ColorBuffer[(ix + launchParams.XOffset) + (long)iy * launchParams.Stride] = bgra;
        }

        private static void XformRaygenChecks(XformLaunchParams launchParams)
        {
            if (OptixGetLaunchIndex.X != 0)
                return;

            // Plain trace - the CH check path performs the transform/vertex checks.
            uint hit = 0, r = 0, g = 0, b = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (XformCheckOriginX, XformCheckOriginY, XformCheckOriginZ),
                (1f, 0f, 0f),
                0.01f,
                100f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref hit, ref r, ref g, ref b);

            // HitObject completion round-trip: traverse without invoking, query the
            // hit object, capture its traverse data, wipe it with a nop, rebuild it
            // via Make, then Invoke - which reruns the CH program (SlotChRuns
            // becomes 2). This exercises OptixHitObject's Make/MakeNop/traverse-data
            // API surface, not SER scheduling itself (see the right tile for that).
            OptixHitObject.Traverse(
                launchParams.Traversable,
                (XformCheckOriginX, XformCheckOriginY, XformCheckOriginZ),
                (1f, 0f, 0f),
                0.01f,
                100f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref hit, ref r, ref g, ref b);

            launchParams.ResultBuffer[SlotSerIsHit] = OptixHitObject.IsHit ? 1u : 0u;
            launchParams.ResultBuffer[SlotSerInstanceId] = OptixHitObject.InstanceId;
            launchParams.ResultBuffer[SlotSerSbtRecordIndex] = OptixHitObject.SbtRecordIndex;

            var serObjectV0 = OptixHitObject.TransformPointFromWorldToObjectSpace(
                new Vector3(2f, 1f, -1f));
            launchParams.ResultBuffer[SlotSerWorldToObjectOk] =
                XformApprox(serObjectV0, new Vector3(0f, 0f, 0f)) ? 1u : 0u;

            ulong gasHandle = OptixHitObject.GASTraversableHandle;
            uint transformCount = OptixHitObject.TransformListSize;
            for (uint i = 0; i < transformCount && i < 2; i++)
                launchParams.TransformHandleScratch[i] =
                    OptixHitObject.GetTransformListHandle(i);
            OptixHitObject.GetTraverseData(out OptixTraverseData traverseData);

            OptixHitObject.MakeNop();
            launchParams.ResultBuffer[SlotSerNopOk] = OptixHitObject.IsNop ? 1u : 0u;

            OptixHitObject.Make(
                gasHandle,
                new Vector3(XformCheckOriginX, XformCheckOriginY, XformCheckOriginZ),
                new Vector3(1f, 0f, 0f),
                0.01f,
                0f,
                OptixRayFlags.DisableAnyHit,
                in traverseData,
                launchParams.TransformHandleScratchAddress,
                transformCount);
            launchParams.ResultBuffer[SlotSerRemadeIsHit] = OptixHitObject.IsHit ? 1u : 0u;

            OptixHitObject.Invoke(ref hit, ref r, ref g, ref b);
        }

        private static void XformMiss(XformLaunchParams launchParams) { }

        private static void XformClosestHit(XformLaunchParams launchParams)
        {
            if (launchParams.CheckMode != 0)
            {
                XformClosestHitChecks(launchParams);
                return;
            }

            // Visual shading path: fetch the object-space vertices, build the
            // object normal, push it through the transform stack, and light it.
            OptixGetTriangleVertexData.CurrentHit(
                out Vector3 v0, out Vector3 v1, out Vector3 v2);

            var edge1 = v1 - v0;
            var edge2 = v2 - v0;
            var objectNormal = Vector3.Cross(edge1, edge2);

            var worldNormal = OptixTransforms.TransformNormalFromObjectToWorldSpace(objectNormal);
            float normalLength = XMath.Sqrt(
                worldNormal.X * worldNormal.X +
                worldNormal.Y * worldNormal.Y +
                worldNormal.Z * worldNormal.Z);
            worldNormal /= XMath.Max(normalLength, 1e-9f);

            var lightDirection = new Vector3(0.5f, 0.7f, 0.5f);
            float ndotl = XMath.Abs(
                worldNormal.X * lightDirection.X +
                worldNormal.Y * lightDirection.Y +
                worldNormal.Z * lightDirection.Z);
            float shade = 0.35f + 0.65f * ndotl;

            var (u, v) = OptixGetTriangleBarycentrics.Value;
            float w = 1f - u - v;
            float cr, cg, cb;
            if (OptixGetInstanceId.Value == XformInstanceId0)
            {
                cr = 0.9f * w + 0.9f * u;
                cg = 0.4f * u + 0.8f * v;
                cb = 0.3f;
            }
            else
            {
                cr = 0.3f;
                cg = 0.5f * w + 0.9f * u;
                cb = 0.9f * v + 0.6f * w;
            }

            OptixPayload.Set(0, 1);
            OptixPayload.Set(1, (uint)(XMath.Clamp(cr * shade, 0f, 1f) * 255f));
            OptixPayload.Set(2, (uint)(XMath.Clamp(cg * shade, 0f, 1f) * 255f));
            OptixPayload.Set(3, (uint)(XMath.Clamp(cb * shade, 0f, 1f) * 255f));
        }

        private static void XformClosestHitChecks(XformLaunchParams launchParams)
        {
            launchParams.ResultBuffer[SlotChRuns]++;
            launchParams.ResultBuffer[SlotHitT] =
                Interop.FloatAsInt(OptixGetRayTmax.Value);

            OptixGetTriangleVertexData.CurrentHit(
                out Vector3 v0, out Vector3 v1, out Vector3 v2);
            launchParams.ResultBuffer[SlotVertexDataOk] =
                XformApprox(v0, new Vector3(0f, 0f, 0f)) &&
                XformApprox(v1, new Vector3(1f, 0f, 0f)) &&
                XformApprox(v2, new Vector3(0f, 1f, 0f)) ? 1u : 0u;

            OptixGetTriangleVertexData.FromHandle(
                OptixGetGASTraversableHandle.Value,
                OptixGetPrimitiveIndex.Value,
                OptixGetSbtGASIndex.Value,
                0f,
                out Vector3 h0, out Vector3 h1, out Vector3 h2);
            launchParams.ResultBuffer[SlotVertexFromHandleOk] =
                XformApprox(h0, v0) && XformApprox(h1, v1) && XformApprox(h2, v2) ? 1u : 0u;

            var w0 = OptixTransforms.TransformPointFromObjectToWorldSpace(v0);
            var w1 = OptixTransforms.TransformPointFromObjectToWorldSpace(v1);
            var w2 = OptixTransforms.TransformPointFromObjectToWorldSpace(v2);
            launchParams.ResultBuffer[SlotObjectToWorldOk] =
                XformApprox(w0, new Vector3(2f, 1f, -1f)) &&
                XformApprox(w1, new Vector3(2f, 1f, -3f)) &&
                XformApprox(w2, new Vector3(2f, 3f, -1f)) ? 1u : 0u;

            var o0 = OptixTransforms.TransformPointFromWorldToObjectSpace(w0);
            launchParams.ResultBuffer[SlotWorldToObjectOk] =
                XformApprox(o0, v0) ? 1u : 0u;

            var worldNormal = OptixTransforms.TransformNormalFromObjectToWorldSpace(
                new Vector3(0f, 0f, 1f));
            launchParams.ResultBuffer[SlotNormalOk] =
                XMath.Abs(worldNormal.X) > 10f * XMath.Abs(worldNormal.Y) &&
                XMath.Abs(worldNormal.X) > 10f * XMath.Abs(worldNormal.Z) ? 1u : 0u;

            launchParams.ResultBuffer[SlotInstanceId] = OptixGetInstanceId.Value;
            launchParams.ResultBuffer[SlotInstanceIndex] = OptixGetInstanceIndex.Value;
            launchParams.ResultBuffer[SlotTransformListSize] = OptixTransformList.Size;
        }

        private static unsafe OptixInstance XformMakeInstance(
            IntPtr gasHandle, uint instanceId, float[] transform)
        {
            var instance = new OptixInstance
            {
                TraversableHandle = unchecked((ulong)gasHandle.ToInt64()),
                InstanceId = instanceId,
                VisibilityMask = 255,
                SbtOffset = 0,
                Flags = 0,
            };
            for (int i = 0; i < 12; i++)
                instance.Transform[i] = transform[i];
            return instance;
        }

        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public int Width;
            public int Height;
            public int XOffset;
            public int Stride;
            public ulong Traversable;
            public float RayTime;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        private struct RadiancePayload
        {
            public uint P0, P1, P2;
        }

        private static void RaygenRenderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            float aspect = (float)launchParams.Width / launchParams.Height;
            float ndcX = (2f * ((ix + 0.5f) / launchParams.Width) - 1f) * aspect;
            float ndcY = 1f - 2f * ((iy + 0.5f) / launchParams.Height);

            float dx = launchParams.ForwardX * launchParams.Focal + launchParams.RightX * ndcX + launchParams.UpX * ndcY;
            float dy = launchParams.ForwardY * launchParams.Focal + launchParams.RightY * ndcX + launchParams.UpY * ndcY;
            float dz = launchParams.ForwardZ * launchParams.Focal + launchParams.RightZ * ndcX + launchParams.UpZ * ndcY;
            float invLen = 1f / XMath.Sqrt(dx * dx + dy * dy + dz * dz);
            dx *= invLen;
            dy *= invLen;
            dz *= invLen;

            uint p0 = 0, p1 = 0, p2 = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                1e-3f,
                1e6f,
                launchParams.RayTime,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref p0,
                ref p1,
                ref p2);

            float r = Interop.IntAsFloat(p0);
            float g = Interop.IntAsFloat(p1);
            float b = Interop.IntAsFloat(p2);

            uint ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
            uint gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
            uint bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);

            uint rgba = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);

            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            launchParams.ColorBuffer[pixel] = rgba;
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.08f);
            OptixPayloadInterop.SetFloat(1, 0.08f);
            OptixPayloadInterop.SetFloat(2, 0.12f);
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.3f);
            OptixPayloadInterop.SetFloat(1, 0.75f);
            OptixPayloadInterop.SetFloat(2, 0.55f);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private const int Width = 1536;
        private const int Height = 768;
        private const int TileWidth = Width / 3;
        private const int TileHeight = Height;
        private const float TranslateDistance = 2.5f;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private BuiltAccelStructure gas;
        private BuiltMotionTransform motionTransform;
        private BuiltAccelStructure ias;
        private LaunchParams launchParams;

        // Zero-copy CUDA-GL interop: all three tiles' pipelines write directly into
        // different XOffset/Stride windows of this ONE full-window buffer every
        // frame (see Rendering/CudaGlInteropDisplayBuffer.cs) - no host round-trip.
        private CudaGlInteropDisplayBuffer interopBuffer;
        private FullscreenQuad quad;

        private OptixLogCallback logCallback;

        // Only cameraYaw/cameraPitch are shared (one mouse drag orbits all three
        // tiles in sync); each tile keeps its own fixed scene center/orbit distance.
        private float cameraYaw;
        private float cameraPitch;
        private float cameraFocal;
        private bool wasLeftMouseDown;
        private float lastMouseX;
        private float lastMouseY;
        private const float MouseSensitivity = 0.3f;

        private bool motionEnabled = true;
        private bool spaceWasDown;
        private readonly System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();

        private int frameCount;

        private RayTracingPipeline<XformLaunchParams> xformPipeline;
        private XformLaunchParams xformLaunchParams;
        private MemoryBuffer1D<uint, Stride1D.Dense> xformResultBuffer;
        private MemoryBuffer1D<ulong, Stride1D.Dense> xformTransformScratch;
        private MemoryBuffer1D<Vector3, Stride1D.Dense> xformVertexBuffer;
        private MemoryBuffer1D<Tri, Stride1D.Dense> xformIndexBuffer;
        private MemoryBuffer1D<OptixInstance, Stride1D.Dense> xformInstanceBuffer;
        private BuiltAccelStructure xformGas;
        private BuiltAccelStructure xformIas;
        private (Vector3 Center, float Radius) motionMeshBounds;
        private static readonly Vector3 XformCenter = new Vector3(0.6f, 1.4f, -0.4f);
        private const float XformOrbitDistance = 7.5f;

        private RayTracingPipeline<SerLaunchParams> serPipeline;
        private SerLaunchParams serLaunchParams;
        private MemoryBuffer1D<uint, Stride1D.Dense> serMismatchBuffer;
        private uint[] serMismatchHost;
        private BuiltAccelStructure serAccel;
        private (Vector3 Center, float Radius) serMeshBounds;
        private long totalSerMismatches;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample17: rigid-body motion blur (matrix motion transform)");
            Console.WriteLine("Initializing CUDA + OptiX (validation mode ALL)...");

            context = ILGPU.Context.Create(b => b.Cuda().EnableAlgorithms());
            accelerator = context.CreateCudaAccelerator(0);

            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            rayTracer = OptixRayTracer.Create(accelerator, new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.All,
            });

            Console.WriteLine("Compiling pipeline (motion-blur-enabled traversable graph)...");
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>(ClosestHitRadiance, AnyHitRadiance))
                // IAS -> matrix motion transform -> GAS is not "single-level
                // instancing" (that flag specifically means IAS directly connected to
                // GAS with no transforms in between) - AllowAny + graph depth 3
                // (IAS, transform, GAS) is what this traversable graph actually is.
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowAny)
                .MaxTraversableGraphDepth(3)
                .UsesMotionBlur()
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            Console.WriteLine("Loading mesh (cow.obj)...");
            var (vertices, indices) = Model.LoadPositionsOnly("models/meshes/cow.obj");
            Console.WriteLine($"Loaded {vertices.Length} vertices, {indices.Length} triangles.");

            using (var vertexBuffer = accelerator.Allocate1D(vertices))
            using (var indexBuffer = accelerator.Allocate1D(indices))
            {
                Console.WriteLine("Building static GAS...");
                gas = new OptixAccelBuilder()
                    .WithDeviceContext(rayTracer.DeviceContext)
                    .WithAccelerator(accelerator)
                    .AddTriangleMesh(vertexBuffer, indexBuffer)
                    .AllowCompaction()
                    .Build();

                Console.WriteLine("Building SER tile's own static GAS (same mesh, no motion transform)...");
                serAccel = new OptixAccelBuilder()
                    .WithDeviceContext(rayTracer.DeviceContext)
                    .WithAccelerator(accelerator)
                    .AddTriangleMesh(vertexBuffer, indexBuffer)
                    .AllowCompaction()
                    .Build();
            }

            Console.WriteLine("Building matrix motion transform (spin + translate)...");
            var key0 = MakeTransform(0f, 0f);
            var key1 = MakeTransform(MathF.PI / 2f, TranslateDistance);
            motionTransform = OptixMotionTransformBuilder.BuildMatrixMotionTransform(
                rayTracer.DeviceContext, accelerator, gas.TraversableHandle,
                timeBegin: 0f, timeEnd: 1f, transformKey0: key0, transformKey1: key1);

            Console.WriteLine("Building IAS (one instance referencing the motion transform)...");
            ias = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .BuildInstanceAccelFromHandles(
                    new[] { motionTransform.TraversableHandle },
                    new uint[] { 0 });

            var (center, radius) = ComputeBounds(vertices);
            serMeshBounds = (center, radius * 1.6f);
            // Center the camera between the two motion-key positions, and widen the
            // orbit distance so the whole translate range stays framed.
            center += new Vector3(TranslateDistance * 0.5f, 0f, 0f);
            motionMeshBounds = (center, radius + TranslateDistance * 0.5f);
            SetupCamera(center, radius + TranslateDistance * 0.5f);

            interopBuffer = new CudaGlInteropDisplayBuffer(Width, Height, accelerator);
            launchParams.XOffset = 0;
            launchParams.Stride = Width;
            launchParams.Traversable = unchecked((ulong)ias.TraversableHandle.ToInt64());

            quad = new FullscreenQuad();

            Console.WriteLine("Motion transform + IAS built successfully: PASS.");

            SetupXformScene();
            SetupSerScene();

            xformLaunchParams.XOffset = TileWidth;
            xformLaunchParams.Stride = Width;
            serLaunchParams.XOffset = TileWidth * 2;
            serLaunchParams.Stride = Width;

            ApplyCameraOrbit();

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit (orbits all three tiles together). " +
                "Left = motion transform, middle = static transforms/HitObject, right = SER. " +
                "Space toggles motion (left tile, bonus - on by default). Esc to quit.");
        }

        private void SetupSerScene()
        {
            Console.WriteLine("Compiling SER pipeline (Traverse/Reorder/Invoke device intrinsics)...");
            serPipeline = rayTracer.CreatePipeline<SerLaunchParams>(b => b
                .Raygen(SerRaygen)
                .RayType("radiance", r => r
                    .Payload<SerRadiancePayload>()
                    .Miss(SerMiss)
                    .HitGroup<byte>(SerClosestHit))
                .MaxTraceDepth(2));
            serPipeline.SetHitRecords<byte>(new byte[] { 0 });

            serMismatchBuffer = accelerator.Allocate1D<uint>(TileWidth * TileHeight);
            serMismatchHost = new uint[TileWidth * TileHeight];
            serLaunchParams.MismatchBuffer = OptixDeviceView<uint>.From(serMismatchBuffer);
            serLaunchParams.Traversable = unchecked((ulong)serAccel.TraversableHandle.ToInt64());
            serLaunchParams.Width = TileWidth;
            serLaunchParams.Height = TileHeight;

            // Startup correctness check: render once and count mismatches between
            // the classic-trace and SER-invoke paths - should be exactly zero.
            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, serMeshBounds.Center, serMeshBounds.Radius);
            serLaunchParams.OriginX = origin.X;
            serLaunchParams.OriginY = origin.Y;
            serLaunchParams.OriginZ = origin.Z;
            serLaunchParams.ForwardX = forward.X;
            serLaunchParams.ForwardY = forward.Y;
            serLaunchParams.ForwardZ = forward.Z;
            serLaunchParams.RightX = right.X;
            serLaunchParams.RightY = right.Y;
            serLaunchParams.RightZ = right.Z;
            serLaunchParams.UpX = up.X;
            serLaunchParams.UpY = up.Y;
            serLaunchParams.UpZ = up.Z;
            serLaunchParams.Focal = cameraFocal;

            // Startup-only throwaway target - the live render loop re-points
            // ColorBuffer/XOffset/Stride at the shared interop buffer below.
            using (var startupCheckBuffer = accelerator.Allocate1D<uint>(TileWidth * TileHeight))
            {
                serLaunchParams.ColorBuffer = OptixDeviceView<uint>.From(startupCheckBuffer);
                serLaunchParams.XOffset = 0;
                serLaunchParams.Stride = TileWidth;
                serPipeline.Launch(serLaunchParams, TileWidth, TileHeight);
                accelerator.Synchronize();
            }
            serMismatchBuffer.CopyToCPU(serMismatchHost);
            long startupMismatches = 0;
            foreach (var m in serMismatchHost)
                startupMismatches += m;
            Console.WriteLine($"SER startup frame: {startupMismatches} of {TileWidth * TileHeight} pixels mismatched between classic-trace and SER-invoke paths.");
            Console.WriteLine(startupMismatches == 0
                ? "Classic optixTrace vs. traverse/reorder/invoke: PASS (pixel-identical)"
                : "Classic optixTrace vs. traverse/reorder/invoke: FAIL (see magenta pixels)");
        }

        private void SetupXformScene()
        {
            Console.WriteLine("Compiling static-transform/vertex-fetch/HitObject pipeline...");
            xformPipeline = rayTracer.CreatePipeline<XformLaunchParams>(b => b
                .Raygen(XformRaygen)
                .RayType("radiance", r => r
                    .Payload<XformRadiancePayload>()
                    .Miss(XformMiss)
                    .HitGroup<byte>(XformClosestHit))
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
            xformPipeline.SetHitRecords<byte>(new byte[] { 0 });

            var xformVertices = new[] { XformV0, XformV1, XformV2 };
            var xformIndices = new[] { new Tri(0, 1, 2) };
            xformVertexBuffer = accelerator.Allocate1D(xformVertices);
            xformIndexBuffer = accelerator.Allocate1D(xformIndices);

            Console.WriteLine("Building GAS (with AllowRandomVertexAccess) + transformed 2-instance IAS...");
            xformGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(xformVertexBuffer, xformIndexBuffer)
                .AllowRandomVertexAccess()
                .Build();

            xformInstanceBuffer = accelerator.Allocate1D(new[]
            {
                XformMakeInstance(xformGas.TraversableHandle, XformInstanceId0, XformInstanceTransform0),
                XformMakeInstance(xformGas.TraversableHandle, XformInstanceId1, XformInstanceTransform1),
            });
            xformIas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddInstances(xformInstanceBuffer, 2)
                .Build();

            xformResultBuffer = accelerator.Allocate1D<uint>(SlotCount);
            xformTransformScratch = accelerator.Allocate1D<ulong>(2);
            xformResultBuffer.MemSetToZero();
            xformTransformScratch.MemSetToZero();

            // ColorBuffer is left unset here - RunXformChecks below only exercises
            // CheckMode=1 (a 1x1 launch that never writes ColorBuffer); OnLoad
            // re-points it at the shared interop buffer before the live loop starts.
            xformLaunchParams = new XformLaunchParams
            {
                ResultBuffer = OptixDeviceView<uint>.From(xformResultBuffer),
                TransformHandleScratch = OptixDeviceView<ulong>.From(xformTransformScratch),
                TransformHandleScratchAddress =
                    unchecked((ulong)xformTransformScratch.NativePtr.ToInt64()),
                Traversable = unchecked((ulong)xformIas.TraversableHandle.ToInt64()),
            };

            RunXformChecks();
        }

        private void RunXformChecks()
        {
            int failures = 0;
            void Check(string name, bool condition, string detail)
            {
                Console.WriteLine($"  {(condition ? "PASS" : "FAIL")}  {name} ({detail})");
                if (!condition)
                    failures++;
            }

            Console.WriteLine("Running 1x1 device-check launch (static transforms/vertex-fetch/HitObject)...");
            xformLaunchParams.CheckMode = 1;
            xformPipeline.Launch(xformLaunchParams, 1, 1);
            accelerator.Synchronize();
            xformLaunchParams.CheckMode = 0;
            var results = xformResultBuffer.GetAsArray1D();

            float hitT = BitConverter.UInt32BitsToSingle(results[SlotHitT]);
            Check("closest-hit ran twice (trace + HitObject invoke)",
                results[SlotChRuns] == 2, $"runs={results[SlotChRuns]}");
            Check("hit distance", Math.Abs(hitT - XformExpectedHitT) < 1e-2f, $"t={hitT}");
            Check("optixGetTriangleVertexData (current hit)",
                results[SlotVertexDataOk] == 1, $"slot={results[SlotVertexDataOk]}");
            Check("optixGetTriangleVertexDataFromHandle",
                results[SlotVertexFromHandleOk] == 1, $"slot={results[SlotVertexFromHandleOk]}");
            Check("object-to-world transform composition",
                results[SlotObjectToWorldOk] == 1, $"slot={results[SlotObjectToWorldOk]}");
            Check("world-to-object transform composition",
                results[SlotWorldToObjectOk] == 1, $"slot={results[SlotWorldToObjectOk]}");
            Check("normal transform (inverse-transpose)",
                results[SlotNormalOk] == 1, $"slot={results[SlotNormalOk]}");
            Check("optixGetInstanceId",
                results[SlotInstanceId] == XformInstanceId0, $"id={results[SlotInstanceId]}");
            Check("optixGetInstanceIndex",
                results[SlotInstanceIndex] == 0, $"index={results[SlotInstanceIndex]}");
            Check("transform list size",
                results[SlotTransformListSize] == 1, $"size={results[SlotTransformListSize]}");
            Check("HitObject traverse hit",
                results[SlotSerIsHit] == 1, $"slot={results[SlotSerIsHit]}");
            Check("HitObject instance id",
                results[SlotSerInstanceId] == XformInstanceId0, $"id={results[SlotSerInstanceId]}");
            Check("HitObject world-to-object transform",
                results[SlotSerWorldToObjectOk] == 1, $"slot={results[SlotSerWorldToObjectOk]}");
            Check("HitObject MakeNop -> IsNop",
                results[SlotSerNopOk] == 1, $"slot={results[SlotSerNopOk]}");
            Check("HitObject Make(traverse data) -> IsHit",
                results[SlotSerRemadeIsHit] == 1, $"slot={results[SlotSerRemadeIsHit]}");

            Console.WriteLine(failures == 0
                ? "Static-transform mode startup checks: ALL PASSED"
                : $"Static-transform mode startup checks: {failures} FAILED");
        }

        // 3x4 row-major object-to-world matrix: rotate angleRadians around Y, then
        // translate along X by translateX.
        private static float[] MakeTransform(float angleRadians, float translateX)
        {
            float c = MathF.Cos(angleRadians);
            float s = MathF.Sin(angleRadians);
            return new[]
            {
                c, 0f, s, translateX,
                0f, 1f, 0f, 0f,
                -s, 0f, c, translateX * 0.15f,
            };
        }

        private static (Vector3 Center, float Radius) ComputeBounds(Vector3[] vertices)
        {
            var min = vertices[0];
            var max = vertices[0];
            foreach (var v in vertices)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            var center = (min + max) * 0.5f;
            var radius = (max - min).Length() * 0.5f;
            return (center, radius);
        }

        private void SetupCamera(Vector3 center, float radius)
        {
            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.6f, 0.5f, 1.4f));
            cameraYaw = MathF.Atan2(initialDir.X, initialDir.Z) * 180f / MathF.PI;
            cameraPitch = MathF.Asin(Math.Clamp(initialDir.Y, -1f, 1f)) * 180f / MathF.PI;

            launchParams.Width = TileWidth;
            launchParams.Height = TileHeight;
        }

        private static (Vector3 Origin, Vector3 Forward, Vector3 Right, Vector3 Up) OrbitBasis(
            float yawDegrees, float pitchDegrees, Vector3 center, float distance)
        {
            float yawRad = yawDegrees * MathF.PI / 180f;
            float pitchRad = pitchDegrees * MathF.PI / 180f;

            var dir = new Vector3(
                MathF.Cos(pitchRad) * MathF.Sin(yawRad),
                MathF.Sin(pitchRad),
                MathF.Cos(pitchRad) * MathF.Cos(yawRad));

            var origin = center + dir * distance;
            var forward = Vector3.Normalize(center - origin);
            var worldUp = new Vector3(0f, 1f, 0f);
            var right = Vector3.Normalize(Vector3.Cross(forward, worldUp));
            var up = Vector3.Cross(right, forward);
            return (origin, forward, right, up);
        }

        // Recomputes all three tiles' camera basis from the shared yaw/pitch - all
        // three tiles always render, so all three always need up-to-date camera params.
        private void ApplyCameraOrbit()
        {
            var (origin, forward, right, up) = OrbitBasis(
                cameraYaw, cameraPitch, motionMeshBounds.Center, motionMeshBounds.Radius * 1.8f);
            launchParams.OriginX = origin.X;
            launchParams.OriginY = origin.Y;
            launchParams.OriginZ = origin.Z;
            launchParams.ForwardX = forward.X;
            launchParams.ForwardY = forward.Y;
            launchParams.ForwardZ = forward.Z;
            launchParams.RightX = right.X;
            launchParams.RightY = right.Y;
            launchParams.RightZ = right.Z;
            launchParams.UpX = up.X;
            launchParams.UpY = up.Y;
            launchParams.UpZ = up.Z;
            launchParams.Focal = cameraFocal;

            var (xOrigin, xForward, xRight, xUp) = OrbitBasis(cameraYaw, cameraPitch, XformCenter, XformOrbitDistance);
            xformLaunchParams.OriginX = xOrigin.X;
            xformLaunchParams.OriginY = xOrigin.Y;
            xformLaunchParams.OriginZ = xOrigin.Z;
            xformLaunchParams.ForwardX = xForward.X;
            xformLaunchParams.ForwardY = xForward.Y;
            xformLaunchParams.ForwardZ = xForward.Z;
            xformLaunchParams.RightX = xRight.X;
            xformLaunchParams.RightY = xRight.Y;
            xformLaunchParams.RightZ = xRight.Z;
            xformLaunchParams.UpX = xUp.X;
            xformLaunchParams.UpY = xUp.Y;
            xformLaunchParams.UpZ = xUp.Z;
            xformLaunchParams.Focal = cameraFocal;

            var (sOrigin, sForward, sRight, sUp) = OrbitBasis(cameraYaw, cameraPitch, serMeshBounds.Center, serMeshBounds.Radius);
            serLaunchParams.OriginX = sOrigin.X;
            serLaunchParams.OriginY = sOrigin.Y;
            serLaunchParams.OriginZ = sOrigin.Z;
            serLaunchParams.ForwardX = sForward.X;
            serLaunchParams.ForwardY = sForward.Y;
            serLaunchParams.ForwardZ = sForward.Z;
            serLaunchParams.RightX = sRight.X;
            serLaunchParams.RightY = sRight.Y;
            serLaunchParams.RightZ = sRight.Z;
            serLaunchParams.UpX = sUp.X;
            serLaunchParams.UpY = sUp.Y;
            serLaunchParams.UpZ = sUp.Z;
            serLaunchParams.Focal = cameraFocal;
        }

        private void UpdateMouseOrbit()
        {
            bool leftDown = MouseState.IsButtonDown(MouseButton.Left);
            float x = MouseState.X;
            float y = MouseState.Y;

            if (leftDown && !wasLeftMouseDown)
            {
                lastMouseX = x;
                lastMouseY = y;
            }
            else if (leftDown)
            {
                float dx = (x - lastMouseX) * MouseSensitivity;
                float dy = (y - lastMouseY) * MouseSensitivity;
                lastMouseX = x;
                lastMouseY = y;

                cameraYaw += dx;
                cameraPitch = Math.Clamp(cameraPitch - dy, -89f, 89f);
                ApplyCameraOrbit();
            }

            wasLeftMouseDown = leftDown;
        }

        protected override void OnUpdateFrame(FrameEventArgs args)
        {
            base.OnUpdateFrame(args);
            if (KeyboardState.IsKeyDown(Keys.Escape))
            {
                Close();
                return;
            }

            bool spaceDown = KeyboardState.IsKeyDown(Keys.Space);
            if (spaceDown && !spaceWasDown)
            {
                motionEnabled = !motionEnabled;
                Console.WriteLine($"[Motion] {(motionEnabled ? "enabled" : "disabled (frozen at key 0)")}");
            }
            spaceWasDown = spaceDown;

            UpdateMouseOrbit();
        }

        protected override unsafe void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            if (motionEnabled)
            {
                // Triangle wave over 4 seconds: 0 -> 1 -> 0, a smooth continuous
                // back-and-forth spin/translate instead of snapping at the loop point.
                float phase = (float)(clock.Elapsed.TotalSeconds % 4.0) / 4f;
                launchParams.RayTime = 1f - MathF.Abs(1f - phase * 2f);
            }
            else
            {
                launchParams.RayTime = 0f;
            }

            var stream = (CudaStream)accelerator.DefaultStream;
            interopBuffer.MapCuda(stream);
            interopBuffer.GetCudaArrayView();
            var fullWindowView = new OptixDeviceView<uint>((uint*)interopBuffer.NativePtr, Width * (long)Height);

            launchParams.ColorBuffer = fullWindowView;
            pipeline.Launch(launchParams, TileWidth, TileHeight);

            xformLaunchParams.ColorBuffer = fullWindowView;
            xformPipeline.Launch(xformLaunchParams, TileWidth, TileHeight);

            serLaunchParams.ColorBuffer = fullWindowView;
            serPipeline.Launch(serLaunchParams, TileWidth, TileHeight);

            accelerator.Synchronize();
            interopBuffer.UnmapCuda(stream);
            interopBuffer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(interopBuffer.GlTextureHandle);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
            {
                serMismatchBuffer.CopyToCPU(serMismatchHost);
                long mismatches = 0;
                foreach (var m in serMismatchHost)
                    mismatches += m;
                totalSerMismatches += mismatches;
                Console.WriteLine($"[Sample17] frame {frameCount}, SER this-frame mismatches: {mismatches}");
            }
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }

        protected override void OnUnload()
        {
            quad?.Dispose();
            interopBuffer?.Dispose();

            xformIas?.Dispose();
            xformGas?.Dispose();
            xformInstanceBuffer?.Dispose();
            xformIndexBuffer?.Dispose();
            xformVertexBuffer?.Dispose();
            xformTransformScratch?.Dispose();
            xformResultBuffer?.Dispose();
            xformPipeline?.Dispose();

            serAccel?.Dispose();
            serMismatchBuffer?.Dispose();
            serPipeline?.Dispose();

            ias?.Dispose();
            motionTransform?.Dispose();
            gas?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            Console.WriteLine($"[Sample17] total SER mismatches observed across the whole session: {totalSerMismatches}");

            base.OnUnload();
        }
    }
}
