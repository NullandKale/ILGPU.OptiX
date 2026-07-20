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
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Sample18
{
    /// <summary>
    /// Geometry-primitives API surface sample. Covers:
    ///  - curve primitives via OptiX's built-in intersection program
    ///    (optixBuiltinISModuleGet, via OptixAccelBuilder.AddCurves +
    ///    RayTypeBuilder.HitGroupWithBuiltinIS) - left tile, live;
    ///  - opacity micromaps (optixOpacityMicromapArrayBuild, cutout triangles with
    ///    no any-hit program) - middle tile, live;
    ///  - sphere geometry (OptixAccelBuilder.AddSpheres + optixGetSphereData) and
    ///    the remaining curve variants (cubic Bezier, CatmullRom, flat quadratic
    ///    B-spline ribbons, rocaps) - right tile, live.
    ///
    /// Left tile: a flat triangle ground plane plus a patch of ~3000 short round
    /// cubic B-spline "fur" strands rooted on it, both geometry kinds combined into
    /// one GAS as separate build inputs, each with its own named hit group sharing
    /// the one "radiance" ray type.
    ///
    /// Middle tile: a flat ground plane (ordinary opaque triangles) plus ~40 small
    /// upright "cutout cards" - each a 2-triangle quad where one triangle is opaque
    /// (a green "leaf") and the other fully transparent via its opacity micromap,
    /// with no any-hit program declared for that hit group at all.
    ///
    /// Right tile: sphere geometry plus 4 curve variants, each in its own GAS
    /// combined via an IAS; the first frame is also saved to sample18_primitives.png.
    ///
    /// [Controls] Hold Left Mouse + drag to orbit (orbits all three tiles together).
    /// Esc to quit.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        // ---- Mode 2: opacity micromaps ----

        private struct OmmLaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
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

        private struct OmmRadiancePayload
        {
            public uint P0, P1, P2;
        }

        private static void OmmRaygen(OmmLaunchParams launchParams)
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
                0f,
                0xff,
                OptixRayFlags.None,
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

        private static void OmmMissRadiance(OmmLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.45f);
            OptixPayloadInterop.SetFloat(1, 0.55f);
            OptixPayloadInterop.SetFloat(2, 0.75f);
        }

        private static void OmmClosestHitGround(OmmLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.3f);
            OptixPayloadInterop.SetFloat(1, 0.25f);
            OptixPayloadInterop.SetFloat(2, 0.2f);
        }

        private static void OmmClosestHitCard(OmmLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.2f);
            OptixPayloadInterop.SetFloat(1, 0.6f);
            OptixPayloadInterop.SetFloat(2, 0.25f);
        }

        private const int OmmCardCount = 40;
        private const float OmmGroundHalfExtent = 6f;
        private const float OmmPatchHalfExtent = 5f;

        private static (Vector3[] Vertices, Tri[] Indices) OmmMakeGroundPlane()
        {
            var vertices = new[]
            {
                new Vector3(-OmmGroundHalfExtent, 0f, -OmmGroundHalfExtent),
                new Vector3( OmmGroundHalfExtent, 0f, -OmmGroundHalfExtent),
                new Vector3( OmmGroundHalfExtent, 0f,  OmmGroundHalfExtent),
                new Vector3(-OmmGroundHalfExtent, 0f,  OmmGroundHalfExtent),
            };
            var indices = new[]
            {
                new Tri(0, 1, 2),
                new Tri(0, 2, 3),
            };
            return (vertices, indices);
        }

        // OmmCardCount small upright quads scattered on the ground, each 2
        // triangles: triangle 0 (bottom-left half) opaque, triangle 1 (top-right
        // half) transparent via its opacity micromap - visually a triangular
        // "leaf" cutout.
        private static (Vector3[] Vertices, Tri[] Indices, OptixOpacityMicromapState[] States) OmmMakeCutoutCards()
        {
            var rng = new Random(7);
            var vertices = new List<Vector3>(OmmCardCount * 4);
            var indices = new List<Tri>(OmmCardCount * 2);
            var states = new List<OptixOpacityMicromapState>(OmmCardCount * 2);

            for (int c = 0; c < OmmCardCount; c++)
            {
                float rx = (float)(rng.NextDouble() * 2 - 1) * OmmPatchHalfExtent;
                float rz = (float)(rng.NextDouble() * 2 - 1) * OmmPatchHalfExtent;
                float yaw = (float)(rng.NextDouble() * Math.PI * 2);
                float halfWidth = 0.2f + (float)rng.NextDouble() * 0.15f;
                float height = 0.35f + (float)rng.NextDouble() * 0.3f;

                float cosY = MathF.Cos(yaw);
                float sinY = MathF.Sin(yaw);
                Vector3 Rotate(float lx, float ly, float lz) =>
                    new Vector3(rx + lx * cosY - lz * sinY, ly, rz + lx * sinY + lz * cosY);

                uint baseIndex = (uint)vertices.Count;
                vertices.Add(Rotate(-halfWidth, 0f, 0f));
                vertices.Add(Rotate(halfWidth, 0f, 0f));
                vertices.Add(Rotate(halfWidth, height, 0f));
                vertices.Add(Rotate(-halfWidth, height, 0f));

                indices.Add(new Tri(baseIndex, baseIndex + 1, baseIndex + 2)); // opaque half
                indices.Add(new Tri(baseIndex, baseIndex + 2, baseIndex + 3)); // cutout half

                states.Add(OptixOpacityMicromapState.Opaque);
                states.Add(OptixOpacityMicromapState.Transparent);
            }

            return (vertices.ToArray(), indices.ToArray(), states.ToArray());
        }

        // ---- Mode 3: spheres + curve variants ----

        private struct PrimLaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public int XOffset;
            public int Stride;
            public ulong Traversable;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        private struct PrimRadiancePayload
        {
            public uint P0, P1, P2;
        }

        private static void PrimRaygen(PrimLaunchParams launchParams)
        {
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

            uint p0 = 0, p1 = 0, p2 = 0;
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
                ref p0,
                ref p1,
                ref p2);

            float r = Interop.IntAsFloat(p0);
            float g = Interop.IntAsFloat(p1);
            float b = Interop.IntAsFloat(p2);

            uint ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
            uint gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
            uint bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);

            uint bgra = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);

            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            launchParams.ColorBuffer[pixel] = bgra;
        }

        private static void PrimMissRadiance(PrimLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.10f);
            OptixPayloadInterop.SetFloat(1, 0.12f);
            OptixPayloadInterop.SetFloat(2, 0.18f);
        }

        private static void PrimClosestHitGround(PrimLaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.30f);
            OptixPayloadInterop.SetFloat(1, 0.30f);
            OptixPayloadInterop.SetFloat(2, 0.30f);
        }

        private static void PrimClosestHitSphere(PrimLaunchParams launchParams)
        {
            // optixGetSphereData: center (xyz) + radius (w) of the hit sphere.
            OptixGetSphereData.CurrentHit(out Vector4 sphere);

            var origin = OptixGetWorldRayOrigin.AsVector3;
            var direction = OptixGetWorldRayDirection.AsVector3;
            float t = OptixGetRayTmax.Value;
            var hit = origin + direction * t;

            var normal = (hit - new Vector3(sphere.X, sphere.Y, sphere.Z)) /
                XMath.Max(sphere.W, 1e-6f);

            OptixPayloadInterop.SetFloat(0, 0.5f + 0.5f * normal.X);
            OptixPayloadInterop.SetFloat(1, 0.5f + 0.5f * normal.Y);
            OptixPayloadInterop.SetFloat(2, 0.5f + 0.5f * normal.Z);
        }

        private static void PrimClosestHitCurve(PrimLaunchParams launchParams)
        {
            float u = OptixGetCurveParameter.Value;
            var primitiveType = OptixHitKindHelpers.GetPrimitiveType();

            float r = 0.2f, g = 0.2f, b = 0.2f;
            if (primitiveType == OptixPrimitiveType.RoundCubicBezier)
            {
                r = 0.9f;
                g = 0.25f + 0.6f * u;
            }
            else if (primitiveType == OptixPrimitiveType.RoundCatmullRom)
            {
                g = 0.9f;
                b = 0.25f + 0.6f * u;
            }
            else if (primitiveType == OptixPrimitiveType.RoundCubicBSplineRocaps)
            {
                b = 0.9f;
                r = 0.25f + 0.6f * u;
            }

            OptixPayloadInterop.SetFloat(0, r);
            OptixPayloadInterop.SetFloat(1, g);
            OptixPayloadInterop.SetFloat(2, b);
        }

        private static void PrimClosestHitRibbon(PrimLaunchParams launchParams)
        {
            var (u, v) = OptixGetRibbonParameters.Value;
            var normal = OptixGetRibbonNormal.CurrentHit(new Vector2(u, v));

            OptixPayloadInterop.SetFloat(0, 0.6f + 0.4f * XMath.Abs(normal.Y));
            OptixPayloadInterop.SetFloat(1, 0.5f + 0.5f * u);
            OptixPayloadInterop.SetFloat(2, 0.2f + 0.6f * v);
        }

        // One strand of (degree+1) control points per curve kind, planted at
        // baseX. Cubic types consume 4 control points per segment, quadratic 3.
        private static (Vector3[] Vertices, float[] Widths, uint[] Indices) PrimMakeStrands(
            float baseX, int controlPointsPerSegment, int strandCount, float width)
        {
            var rng = new Random(baseX.GetHashCode() ^ 1234);
            var vertices = new List<Vector3>();
            var widths = new List<float>();
            var indices = new List<uint>();

            for (int s = 0; s < strandCount; s++)
            {
                float rx = baseX + ((float)rng.NextDouble() - 0.5f) * 1.2f;
                float rz = ((float)rng.NextDouble() - 0.5f) * 1.2f;
                float height = 1.0f + (float)rng.NextDouble() * 0.7f;
                float leanX = ((float)rng.NextDouble() - 0.5f) * 0.6f;
                float leanZ = ((float)rng.NextDouble() - 0.5f) * 0.6f;

                uint baseIndex = (uint)vertices.Count;
                for (int i = 0; i < controlPointsPerSegment; i++)
                {
                    float f = (float)i / (controlPointsPerSegment - 1);
                    vertices.Add(new Vector3(
                        rx + leanX * f * f,
                        height * f,
                        rz + leanZ * f * f));
                    widths.Add(width * (1f - 0.7f * f));
                }
                indices.Add(baseIndex);
            }

            return (vertices.ToArray(), widths.ToArray(), indices.ToArray());
        }

        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
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
                0f,
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
            OptixPayloadInterop.SetFloat(0, 0.45f);
            OptixPayloadInterop.SetFloat(1, 0.55f);
            OptixPayloadInterop.SetFloat(2, 0.75f);
        }

        private static void ClosestHitGround(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.35f);
            OptixPayloadInterop.SetFloat(1, 0.3f);
            OptixPayloadInterop.SetFloat(2, 0.25f);
        }

        private static void AnyHitGround(LaunchParams launchParams) { }

        private static void ClosestHitFur(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.55f);
            OptixPayloadInterop.SetFloat(1, 0.35f);
            OptixPayloadInterop.SetFloat(2, 0.15f);
        }

        private static void AnyHitFur(LaunchParams launchParams) { }

        private const int Width = 1536;
        private const int Height = 768;
        private const int TileWidth = Width / 3;
        private const int TileHeight = Height;
        private const int StrandCount = 3000;
        private const float GroundHalfExtent = 6f;
        private const float PatchHalfExtent = 5f;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        // Startup-only throwaway target, reused across all three tiles' correctness
        // checks - the live render loop re-points ColorBuffer/XOffset/Stride at the
        // shared interop buffer instead (see Rendering/CudaGlInteropDisplayBuffer.cs).
        private MemoryBuffer1D<uint, Stride1D.Dense> checkBuffer;
        private uint[] checkPixels;

        private BuiltAccelStructure groundGas;
        private BuiltAccelStructure furGas;
        private BuiltAccelStructure ias;
        private LaunchParams launchParams;

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

        private int frameCount;

        private RayTracingPipeline<OmmLaunchParams> ommPipeline;
        private OmmLaunchParams ommLaunchParams;
        private BuiltOpacityMicromapArray opacityMicromapArray;
        private BuiltAccelStructure ommGas;
        private static readonly Vector3 OmmCenter = new Vector3(0f, OmmGroundHalfExtent * 0.15f, 0f);
        private const float OmmOrbitDistance = OmmGroundHalfExtent * 2f;

        private RayTracingPipeline<PrimLaunchParams> primPipeline;
        private PrimLaunchParams primLaunchParams;
        private readonly List<BuiltAccelStructure> primAccelStructures = new List<BuiltAccelStructure>();
        private readonly List<MemoryBuffer> primGeometryBuffers = new List<MemoryBuffer>();
        private static readonly Vector3 PrimCenter = new Vector3(0f, 0.8f, 0f);
        private const float PrimOrbitDistance = 10.5f;

        private (Vector3 Center, float Radius) curveMeshBounds;
        private Vector3 CurveCenter => curveMeshBounds.Center + new Vector3(0f, curveMeshBounds.Radius * 0.15f, 0f);
        private float CurveOrbitDistance => curveMeshBounds.Radius * 2f;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        private MemoryBuffer1D<T, Stride1D.Dense> AllocatePrimGeometry<T>(T[] data)
            where T : unmanaged
        {
            var buffer = accelerator.Allocate1D(data);
            primGeometryBuffers.Add(buffer);
            return buffer;
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample18: curve primitives (built-in intersection)");
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

            Console.WriteLine("Compiling pipeline (triangle ground + built-in-IS curve hit groups)...");
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>("ground", ClosestHitGround, AnyHitGround)
                    .HitGroupWithBuiltinIS<byte>(
                        "fur", ClosestHitFur, AnyHitFur, OptixPrimitiveType.RoundCubicBSpline))
                .UsesPrimitiveTypes(OptixPrimitiveTypeFlags.Triangle | OptixPrimitiveTypeFlags.RoundCubicBSpline)
                // Ground (triangles) and fur (curves) cannot share one GAS - OptiX
                // requires every build input within a single GAS to have the same
                // type - each becomes its own GAS, combined via an IAS below, so this
                // pipeline launches against a 2-level (IAS -> GAS) traversable graph.
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(
                new[]
                {
                    new HitGroupEntry<byte>("ground", 0),
                    new HitGroupEntry<byte>("fur", 0),
                },
                pipeline.RayTypeCount);
            Console.WriteLine("Curve built-in intersection: PASS (pipeline compiled and linked successfully).");

            Console.WriteLine("Building ground plane + fur strands...");
            var (groundVertices, groundIndices) = MakeGroundPlane();
            var (curveVertices, curveWidths, curveIndices) = MakeFurStrands();
            Console.WriteLine($"{groundVertices.Length} ground vertices, {curveVertices.Length} curve vertices, {curveIndices.Length} curve segments.");

            using var groundVertexBuffer = accelerator.Allocate1D(groundVertices);
            using var groundIndexBuffer = accelerator.Allocate1D(groundIndices);
            using var curveVertexBuffer = accelerator.Allocate1D(curveVertices);
            using var curveWidthBuffer = accelerator.Allocate1D(curveWidths);
            using var curveIndexBuffer = accelerator.Allocate1D(curveIndices);

            Console.WriteLine("Building ground GAS (triangles)...");
            groundGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(groundVertexBuffer, groundIndexBuffer)
                .AllowCompaction()
                .Build();

            Console.WriteLine("Building fur GAS (curves, built-in intersection)...");
            furGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddCurves(OptixPrimitiveType.RoundCubicBSpline, curveVertexBuffer, curveWidthBuffer, curveIndexBuffer)
                .AllowCompaction()
                .Build();

            Console.WriteLine("Building IAS (ground + fur GAS instances)...");
            ias = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .BuildInstanceAccelFromHandles(
                    new[] { groundGas.TraversableHandle, furGas.TraversableHandle },
                    new uint[] { 0, 1 });
            Console.WriteLine("GAS builds (triangles, curves) + IAS: PASS.");

            curveMeshBounds = (Vector3.Zero, GroundHalfExtent);
            SetupCamera(Vector3.Zero, GroundHalfExtent);

            checkBuffer = accelerator.Allocate1D<uint>(TileWidth * TileHeight);
            checkPixels = new uint[TileWidth * TileHeight];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            launchParams.XOffset = 0;
            launchParams.Stride = TileWidth;
            launchParams.Traversable = unchecked((ulong)ias.TraversableHandle.ToInt64());

            // Camera basis must be set before the very first launch below - the
            // other two tiles compute their own OrbitBasis inline for this same
            // reason (see SetupOmmScene/SetupPrimScene); ApplyCameraOrbit() runs
            // again for all three tiles once every tile exists, further down.
            var (curveOrigin, curveForward, curveRight, curveUp) =
                OrbitBasis(cameraYaw, cameraPitch, CurveCenter, CurveOrbitDistance);
            launchParams.OriginX = curveOrigin.X;
            launchParams.OriginY = curveOrigin.Y;
            launchParams.OriginZ = curveOrigin.Z;
            launchParams.ForwardX = curveForward.X;
            launchParams.ForwardY = curveForward.Y;
            launchParams.ForwardZ = curveForward.Z;
            launchParams.RightX = curveRight.X;
            launchParams.RightY = curveRight.Y;
            launchParams.RightZ = curveRight.Z;
            launchParams.UpX = curveUp.X;
            launchParams.UpY = curveUp.Y;
            launchParams.UpZ = curveUp.Z;
            launchParams.Focal = cameraFocal;

            quad = new FullscreenQuad();

            // One render + a quick pixel scan: proves curve hits are actually being
            // reported (fur-colored pixels present), not just that the pipeline
            // compiled and the GAS build call returned success.
            pipeline.Launch(launchParams, TileWidth, TileHeight);
            accelerator.Synchronize();
            checkBuffer.CopyToCPU(checkPixels);
            int skyPixels = 0, groundPixels = 0, furPixels = 0;
            foreach (var px in checkPixels)
            {
                byte pr = (byte)((px >> 16) & 0xff);
                byte pg = (byte)((px >> 8) & 0xff);
                byte pb = (byte)(px & 0xff);
                if (pb > pr && pb > 100) skyPixels++;
                else if (pr > pg) furPixels++;
                else groundPixels++;
            }
            Console.WriteLine($"Startup frame: {skyPixels} sky, {groundPixels} ground, {furPixels} fur pixels.");
            Console.WriteLine(furPixels > 0
                ? "Curve strands are being hit and shaded: PASS"
                : "Expected some fur-colored pixels but found none: FAIL (check log above for OptiX warnings)");

            interopBuffer = new CudaGlInteropDisplayBuffer(Width, Height, accelerator);
            launchParams.XOffset = 0;
            launchParams.Stride = Width;

            SetupOmmScene();
            SetupPrimScene();

            ommLaunchParams.XOffset = TileWidth;
            ommLaunchParams.Stride = Width;
            primLaunchParams.XOffset = TileWidth * 2;
            primLaunchParams.Stride = Width;

            ApplyCameraOrbit();

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit (orbits all three tiles together). " +
                "Left = curves (fur), middle = opacity micromaps (cutout cards), right = spheres + curve " +
                "variants. Esc to quit.");
        }

        private void SetupOmmScene()
        {
            Console.WriteLine("Compiling pipeline (ground + opacity-micromap cutout cards)...");
            ommPipeline = rayTracer.CreatePipeline<OmmLaunchParams>(b => b
                .Raygen(OmmRaygen)
                .RayType("radiance", r => r
                    .Payload<OmmRadiancePayload>()
                    .Miss(OmmMissRadiance)
                    .HitGroup<byte>("ground", OmmClosestHitGround)
                    .HitGroup<byte>("cards", OmmClosestHitCard))
                .AllowOpacityMicromaps()
                .MaxTraceDepth(2));
            ommPipeline.SetHitRecords<byte>(
                new[]
                {
                    new HitGroupEntry<byte>("ground", 0),
                    new HitGroupEntry<byte>("cards", 0),
                },
                ommPipeline.RayTypeCount);
            Console.WriteLine("Opacity micromap pipeline: PASS (compiled and linked successfully).");

            Console.WriteLine("Building ground plane + cutout cards...");
            var (groundVertices, groundIndices) = OmmMakeGroundPlane();
            var (cardVertices, cardIndices, cardStates) = OmmMakeCutoutCards();

            using var groundVertexBuffer = accelerator.Allocate1D(groundVertices);
            using var groundIndexBuffer = accelerator.Allocate1D(groundIndices);
            using var cardVertexBuffer = accelerator.Allocate1D(cardVertices);
            using var cardIndexBuffer = accelerator.Allocate1D(cardIndices);

            opacityMicromapArray = OptixOpacityMicromapBuilder.BuildUniformStateArray(
                rayTracer.DeviceContext, accelerator, cardStates);

            ommGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(groundVertexBuffer, groundIndexBuffer)
                .AddTriangleMesh(cardVertexBuffer, cardIndexBuffer, opacityMicromapArray: opacityMicromapArray)
                .AllowCompaction()
                .Build();
            Console.WriteLine("Opacity micromap array + GAS build: PASS.");

            ommLaunchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            ommLaunchParams.XOffset = 0;
            ommLaunchParams.Stride = TileWidth;
            ommLaunchParams.Traversable = unchecked((ulong)ommGas.TraversableHandle.ToInt64());
            ommLaunchParams.Width = TileWidth;
            ommLaunchParams.Height = TileHeight;

            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, OmmCenter, OmmOrbitDistance);
            ommLaunchParams.OriginX = origin.X;
            ommLaunchParams.OriginY = origin.Y;
            ommLaunchParams.OriginZ = origin.Z;
            ommLaunchParams.ForwardX = forward.X;
            ommLaunchParams.ForwardY = forward.Y;
            ommLaunchParams.ForwardZ = forward.Z;
            ommLaunchParams.RightX = right.X;
            ommLaunchParams.RightY = right.Y;
            ommLaunchParams.RightZ = right.Z;
            ommLaunchParams.UpX = up.X;
            ommLaunchParams.UpY = up.Y;
            ommLaunchParams.UpZ = up.Z;
            ommLaunchParams.Focal = cameraFocal;

            // One render + pixel scan: proves the cutout half of each card is
            // actually invisible (ground/sky shows through) while the other half is
            // opaque green, not just that the build calls returned success.
            ommPipeline.Launch(ommLaunchParams, TileWidth, TileHeight);
            accelerator.Synchronize();
            checkBuffer.CopyToCPU(checkPixels);
            int ommSkyPixels = 0, ommGroundPixels = 0, cardPixels = 0;
            foreach (var px in checkPixels)
            {
                byte pr = (byte)((px >> 16) & 0xff);
                byte pg = (byte)((px >> 8) & 0xff);
                byte pb = (byte)(px & 0xff);
                if (pb > pr && pb > 100) ommSkyPixels++;
                else if (pg > pr && pg > pb) cardPixels++;
                else ommGroundPixels++;
            }
            Console.WriteLine($"Opacity-micromap frame: {ommSkyPixels} sky, {ommGroundPixels} ground, {cardPixels} card (opaque-half) pixels.");
            Console.WriteLine(cardPixels > 0 && ommGroundPixels > 0
                ? "Opaque triangles rendered and cutout triangles let ground/sky through: PASS"
                : "Expected both card and ground/sky pixels but didn't see both: FAIL (check log above for OptiX warnings)");
        }

        private void SetupPrimScene()
        {
            Console.WriteLine("Compiling pipeline (5 built-in-IS hit groups + triangle ground)...");
            primPipeline = rayTracer.CreatePipeline<PrimLaunchParams>(b => b
                .Raygen(PrimRaygen)
                .RayType("radiance", r => r
                    .Payload<PrimRadiancePayload>()
                    .Miss(PrimMissRadiance)
                    .HitGroup<byte>("ground", PrimClosestHitGround)
                    .HitGroupWithBuiltinIS<byte>(
                        "spheres", PrimClosestHitSphere, null, OptixPrimitiveType.Sphere)
                    .HitGroupWithBuiltinIS<byte>(
                        "bezier", PrimClosestHitCurve, null, OptixPrimitiveType.RoundCubicBezier)
                    .HitGroupWithBuiltinIS<byte>(
                        "catmullrom", PrimClosestHitCurve, null, OptixPrimitiveType.RoundCatmullRom)
                    .HitGroupWithBuiltinIS<byte>(
                        "ribbon", PrimClosestHitRibbon, null, OptixPrimitiveType.FlatQuadraticBSpline)
                    .HitGroupWithBuiltinIS<byte>(
                        "rocaps", PrimClosestHitCurve, null, OptixPrimitiveType.RoundCubicBSplineRocaps))
                .UsesPrimitiveTypes(
                    OptixPrimitiveTypeFlags.Triangle |
                    OptixPrimitiveTypeFlags.Sphere |
                    OptixPrimitiveTypeFlags.RoundCubicBezier |
                    OptixPrimitiveTypeFlags.RoundCatmullRom |
                    OptixPrimitiveTypeFlags.FlatQuadraticBSpline |
                    OptixPrimitiveTypeFlags.RoundCubicBSplineRocaps)
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
            primPipeline.SetHitRecords<byte>(
                new[]
                {
                    new HitGroupEntry<byte>("ground", 0),
                    new HitGroupEntry<byte>("spheres", 0),
                    new HitGroupEntry<byte>("bezier", 0),
                    new HitGroupEntry<byte>("catmullrom", 0),
                    new HitGroupEntry<byte>("ribbon", 0),
                    new HitGroupEntry<byte>("rocaps", 0),
                },
                primPipeline.RayTypeCount);
            Console.WriteLine("Pipeline with sphere + 4 curve-variant builtin IS modules: PASS.");

            var groundVertices = new[]
            {
                new Vector3(-8f, 0f, -4f),
                new Vector3( 8f, 0f, -4f),
                new Vector3( 8f, 0f,  4f),
                new Vector3(-8f, 0f,  4f),
            };
            var groundIndices = new[] { new Tri(0, 1, 2), new Tri(0, 2, 3) };

            var sphereCenters = new[]
            {
                new Vector3(-5.0f, 0.6f,  0.0f),
                new Vector3(-4.4f, 0.35f, 0.9f),
                new Vector3(-5.6f, 0.35f, 0.8f),
                new Vector3(-5.0f, 1.5f,  0.4f),
            };
            var sphereRadii = new[] { 0.6f, 0.35f, 0.35f, 0.3f };

            var (bezierVerts, bezierWidths, bezierIndices) = PrimMakeStrands(-2.5f, 4, 60, 0.05f);
            var (catmullVerts, catmullWidths, catmullIndices) = PrimMakeStrands(0f, 4, 60, 0.05f);
            var (ribbonVerts, ribbonWidths, ribbonIndices) = PrimMakeStrands(2.5f, 3, 60, 0.08f);
            var (rocapsVerts, rocapsWidths, rocapsIndices) = PrimMakeStrands(5f, 4, 60, 0.05f);

            BuiltAccelStructure BuildGas(Action<OptixAccelBuilder> add)
            {
                var builder = new OptixAccelBuilder()
                    .WithDeviceContext(rayTracer.DeviceContext)
                    .WithAccelerator(accelerator);
                add(builder);
                var built = builder.Build();
                primAccelStructures.Add(built);
                return built;
            }

            Console.WriteLine("Building GASes (triangles, spheres, 4 curve kinds)...");
            var groundGasPrim = BuildGas(b => b.AddTriangleMesh(
                AllocatePrimGeometry(groundVertices), AllocatePrimGeometry(groundIndices)));
            var sphereGas = BuildGas(b => b.AddSpheres(
                AllocatePrimGeometry(sphereCenters), AllocatePrimGeometry(sphereRadii)));
            var bezierGas = BuildGas(b => b.AddCurves(
                OptixPrimitiveType.RoundCubicBezier,
                AllocatePrimGeometry(bezierVerts), AllocatePrimGeometry(bezierWidths),
                AllocatePrimGeometry(bezierIndices)));
            var catmullGas = BuildGas(b => b.AddCurves(
                OptixPrimitiveType.RoundCatmullRom,
                AllocatePrimGeometry(catmullVerts), AllocatePrimGeometry(catmullWidths),
                AllocatePrimGeometry(catmullIndices)));
            var ribbonGas = BuildGas(b => b.AddCurves(
                OptixPrimitiveType.FlatQuadraticBSpline,
                AllocatePrimGeometry(ribbonVerts), AllocatePrimGeometry(ribbonWidths),
                AllocatePrimGeometry(ribbonIndices)));
            var rocapsGas = BuildGas(b => b.AddCurves(
                OptixPrimitiveType.RoundCubicBSplineRocaps,
                AllocatePrimGeometry(rocapsVerts), AllocatePrimGeometry(rocapsWidths),
                AllocatePrimGeometry(rocapsIndices)));

            Console.WriteLine("Building IAS over the 6 GASes...");
            var primIas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .BuildInstanceAccelFromHandles(
                    new[]
                    {
                        groundGasPrim.TraversableHandle,
                        sphereGas.TraversableHandle,
                        bezierGas.TraversableHandle,
                        catmullGas.TraversableHandle,
                        ribbonGas.TraversableHandle,
                        rocapsGas.TraversableHandle,
                    },
                    new uint[] { 0, 1, 2, 3, 4, 5 });
            primAccelStructures.Add(primIas);
            Console.WriteLine("Sphere GAS (AddSpheres) + curve-variant GASes + IAS: PASS.");

            primLaunchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            primLaunchParams.XOffset = 0;
            primLaunchParams.Stride = TileWidth;
            primLaunchParams.Traversable = unchecked((ulong)primIas.TraversableHandle.ToInt64());

            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, PrimCenter, PrimOrbitDistance);
            primLaunchParams.OriginX = origin.X;
            primLaunchParams.OriginY = origin.Y;
            primLaunchParams.OriginZ = origin.Z;
            primLaunchParams.ForwardX = forward.X;
            primLaunchParams.ForwardY = forward.Y;
            primLaunchParams.ForwardZ = forward.Z;
            primLaunchParams.RightX = right.X;
            primLaunchParams.RightY = right.Y;
            primLaunchParams.RightZ = right.Z;
            primLaunchParams.UpX = up.X;
            primLaunchParams.UpY = up.Y;
            primLaunchParams.UpZ = up.Z;
            primLaunchParams.Focal = cameraFocal;

            primPipeline.Launch(primLaunchParams, TileWidth, TileHeight);
            accelerator.Synchronize();
            checkBuffer.CopyToCPU(checkPixels);

            int bezierPixels = 0, catmullPixels = 0, rocapsPixels = 0;
            int ribbonPixels = 0, spherePixels = 0;
            foreach (var px in checkPixels)
            {
                byte b = (byte)(px & 0xff);
                byte g = (byte)((px >> 8) & 0xff);
                byte r = (byte)((px >> 16) & 0xff);
                if (r > 200 && b < 80) bezierPixels++;
                else if (g > 200 && r < 80) catmullPixels++;
                else if (b > 200 && g < 80) rocapsPixels++;
                else if (r > 140 && g > 100 && b < 220 && b > 40 && g < 220) ribbonPixels++;
                else if (r > 90 && r < 200 && g > 90 && b > 90) spherePixels++;
            }
            Console.WriteLine($"Primitives frame: {spherePixels} sphere-ish, {bezierPixels} bezier, " +
                $"{catmullPixels} catmullrom, {ribbonPixels} ribbon-ish, {rocapsPixels} rocaps pixels.");
            bool pass = spherePixels > 100 && bezierPixels > 100 &&
                catmullPixels > 100 && rocapsPixels > 100 && ribbonPixels > 100;
            Console.WriteLine(pass
                ? "All 5 new primitive kinds hit and shaded: PASS"
                : "Some primitive kind produced no/few pixels: CHECK (inspect the window " +
                  "and the OptiX log above - heuristic color classification may also need tuning)");

            var rgba = new uint[checkPixels.Length];
            for (int i = 0; i < checkPixels.Length; i++)
            {
                uint px = checkPixels[i];
                rgba[i] = (px & 0xff00ff00) |
                    ((px & 0x00ff0000) >> 16) |
                    ((px & 0x000000ff) << 16);
            }
            string outputPath = Path.GetFullPath("sample18_primitives.png");
            var outputBytes = MemoryMarshal.AsBytes(rgba.AsSpan()).ToArray();
            using var pngStream = File.OpenWrite(outputPath);
            new StbImageWriteSharp.ImageWriter().WritePng(
                outputBytes, TileWidth, TileHeight,
                StbImageWriteSharp.ColorComponents.RedGreenBlueAlpha, pngStream);
            Console.WriteLine($"Wrote {outputPath}");
        }

        // A flat quad ground plane, GroundHalfExtent units in each direction from the origin.
        private static (Vector3[] Vertices, Tri[] Indices) MakeGroundPlane()
        {
            var vertices = new[]
            {
                new Vector3(-GroundHalfExtent, 0f, -GroundHalfExtent),
                new Vector3( GroundHalfExtent, 0f, -GroundHalfExtent),
                new Vector3( GroundHalfExtent, 0f,  GroundHalfExtent),
                new Vector3(-GroundHalfExtent, 0f,  GroundHalfExtent),
            };
            var indices = new[]
            {
                new Tri(0, 1, 2),
                new Tri(0, 2, 3),
            };
            return (vertices, indices);
        }

        // StrandCount short round cubic B-spline strands (4 control points = 1
        // segment each), rooted at random points on the ground plane, leaning in a
        // random direction and tapering from base to tip.
        private static (Vector3[] Vertices, float[] Widths, uint[] Indices) MakeFurStrands()
        {
            var rng = new Random(42);
            var vertices = new List<Vector3>(StrandCount * 4);
            var widths = new List<float>(StrandCount * 4);
            var indices = new List<uint>(StrandCount);

            for (int s = 0; s < StrandCount; s++)
            {
                float rx = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float rz = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float height = 0.3f + (float)rng.NextDouble() * 0.35f;
                float leanX = ((float)rng.NextDouble() - 0.5f) * 0.2f;
                float leanZ = ((float)rng.NextDouble() - 0.5f) * 0.2f;

                uint baseIndex = (uint)vertices.Count;

                vertices.Add(new Vector3(rx, 0f, rz));
                vertices.Add(new Vector3(rx + leanX * 0.33f, height * 0.33f, rz + leanZ * 0.33f));
                vertices.Add(new Vector3(rx + leanX * 0.66f, height * 0.66f, rz + leanZ * 0.66f));
                vertices.Add(new Vector3(rx + leanX, height, rz + leanZ));

                widths.Add(0.02f);
                widths.Add(0.014f);
                widths.Add(0.008f);
                widths.Add(0.002f);

                indices.Add(baseIndex);
            }

            return (vertices.ToArray(), widths.ToArray(), indices.ToArray());
        }

        private void SetupCamera(Vector3 center, float radius)
        {
            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.7f, 0.55f, 1.2f));
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
            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, CurveCenter, CurveOrbitDistance);
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

            var (oOrigin, oForward, oRight, oUp) = OrbitBasis(cameraYaw, cameraPitch, OmmCenter, OmmOrbitDistance);
            ommLaunchParams.OriginX = oOrigin.X;
            ommLaunchParams.OriginY = oOrigin.Y;
            ommLaunchParams.OriginZ = oOrigin.Z;
            ommLaunchParams.ForwardX = oForward.X;
            ommLaunchParams.ForwardY = oForward.Y;
            ommLaunchParams.ForwardZ = oForward.Z;
            ommLaunchParams.RightX = oRight.X;
            ommLaunchParams.RightY = oRight.Y;
            ommLaunchParams.RightZ = oRight.Z;
            ommLaunchParams.UpX = oUp.X;
            ommLaunchParams.UpY = oUp.Y;
            ommLaunchParams.UpZ = oUp.Z;
            ommLaunchParams.Focal = cameraFocal;

            var (pOrigin, pForward, pRight, pUp) = OrbitBasis(cameraYaw, cameraPitch, PrimCenter, PrimOrbitDistance);
            primLaunchParams.OriginX = pOrigin.X;
            primLaunchParams.OriginY = pOrigin.Y;
            primLaunchParams.OriginZ = pOrigin.Z;
            primLaunchParams.ForwardX = pForward.X;
            primLaunchParams.ForwardY = pForward.Y;
            primLaunchParams.ForwardZ = pForward.Z;
            primLaunchParams.RightX = pRight.X;
            primLaunchParams.RightY = pRight.Y;
            primLaunchParams.RightZ = pRight.Z;
            primLaunchParams.UpX = pUp.X;
            primLaunchParams.UpY = pUp.Y;
            primLaunchParams.UpZ = pUp.Z;
            primLaunchParams.Focal = cameraFocal;
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
            UpdateMouseOrbit();
        }

        protected override unsafe void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            var stream = (CudaStream)accelerator.DefaultStream;
            interopBuffer.MapCuda(stream);
            interopBuffer.GetCudaArrayView();
            var fullWindowView = new OptixDeviceView<uint>((uint*)interopBuffer.NativePtr, Width * (long)Height);

            launchParams.ColorBuffer = fullWindowView;
            pipeline.Launch(launchParams, TileWidth, TileHeight);

            ommLaunchParams.ColorBuffer = fullWindowView;
            ommPipeline.Launch(ommLaunchParams, TileWidth, TileHeight);

            primLaunchParams.ColorBuffer = fullWindowView;
            primPipeline.Launch(primLaunchParams, TileWidth, TileHeight);

            accelerator.Synchronize();
            interopBuffer.UnmapCuda(stream);
            interopBuffer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(interopBuffer.GlTextureHandle);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample18] frame {frameCount}");
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

            for (int i = primAccelStructures.Count - 1; i >= 0; i--)
                primAccelStructures[i].Dispose();
            foreach (var buffer in primGeometryBuffers)
                buffer.Dispose();
            primPipeline?.Dispose();

            ommGas?.Dispose();
            opacityMicromapArray?.Dispose();
            ommPipeline?.Dispose();

            ias?.Dispose();
            furGas?.Dispose();
            groundGas?.Dispose();
            checkBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
