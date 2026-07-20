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

namespace Sample15
{
    /// <summary>
    /// Device context &amp; acceleration-structure API surface sample. Covers:
    ///  - relocation (optixAccelGetRelocationInfo / optixCheckRelocationCompatibility /
    ///    optixAccelRelocate) and the Aabbs emitted property (optixAccelEmitProperty) -
    ///    left tile, live;
    ///  - cluster acceleration structures (CLAS-from-triangles -> GAS-from-CLAS),
    ///    colored per optixGetClusterId, on drivers that support them - right tile,
    ///    live;
    ///  - device properties (optixDeviceContextGetProperty) and the disk cache
    ///    controls (enable/location/database-size round-trip), printed at startup;
    ///  - task-based module compilation (optixModuleCreateWithTasks) - both tiles'
    ///    pipelines compile their raygen module this way by default, not as a
    ///    separate thing to opt into.
    ///
    /// At startup: builds one GAS for a real mesh (cow.obj), checks the emitted AABB
    /// against the mesh's own known bounds, renders it once, relocates the GAS into a
    /// freshly allocated buffer, renders again, and checks the two renders are
    /// pixel-identical - proving the relocated structure traces exactly like the
    /// original. Then prints every OptixDeviceProperty, round-trips the disk cache
    /// controls, and (if the driver supports ClusterAccel) builds a 2x2 grid of CLASes
    /// into a clustered GAS. Both tiles render live every frame - no mode switch.
    ///
    /// [Controls] Hold Left Mouse + drag to orbit (orbits both tiles together). Esc to
    /// quit.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private const uint BaseClusterId = 10;

        private struct ClusterLaunchParams
        {
            // ColorBuffer spans the whole window (shared with the other tile's
            // pipeline, both write into the SAME zero-copy CUDA-GL buffer every
            // frame - no per-tile buffer, no host round-trip); XOffset/Stride place
            // this tile's TileWidth x TileHeight launch at its column.
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

        private struct ClusterRadiancePayload
        {
            public uint Hit, R, G, B;
        }

        private static void ClusterRaygen(ClusterLaunchParams launchParams)
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
                uint shade = (uint)(30f + 50f * iy / height);
                bgra = 0xff000000 | (shade + 30) | (shade << 8) | (shade << 16);
            }

            launchParams.ColorBuffer[(ix + launchParams.XOffset) + (long)iy * launchParams.Stride] = bgra;
        }

        private static void ClusterMiss(ClusterLaunchParams launchParams) { }

        private static void ClusterClosestHit(ClusterLaunchParams launchParams)
        {
            // Color per user-defined cluster id (optixGetClusterId; each quad of
            // the grid is its own CLAS with ids BaseClusterId..BaseClusterId+3).
            uint clusterId = OptixGetClusterId.Value;
            var (u, v) = OptixGetTriangleBarycentrics.Value;
            float tint = 0.55f + 0.45f * (1f - u - v);

            float r = 0.25f, g = 0.25f, b = 0.25f;
            uint local = clusterId - BaseClusterId;
            if (clusterId == OptixGetClusterId.Invalid) { r = 1f; g = 0f; b = 1f; }
            else if (local == 0) { r = 0.95f; g = 0.40f; b = 0.30f; }
            else if (local == 1) { r = 0.35f; g = 0.85f; b = 0.35f; }
            else if (local == 2) { r = 0.35f; g = 0.55f; b = 0.95f; }
            else { r = 0.95f; g = 0.85f; b = 0.35f; }

            OptixPayload.Set(0, 1);
            OptixPayload.Set(1, (uint)(XMath.Clamp(r * tint, 0f, 1f) * 255f));
            OptixPayload.Set(2, (uint)(XMath.Clamp(g * tint, 0f, 1f) * 255f));
            OptixPayload.Set(3, (uint)(XMath.Clamp(b * tint, 0f, 1f) * 255f));
        }

        private struct LaunchParams
        {
            // ColorBuffer/XOffset/Stride: see ClusterLaunchParams' doc comment -
            // same shared-buffer-plus-column-offset scheme.
            public OptixDeviceView<uint> ColorBuffer;
            public int XOffset;
            public int Stride;
            public int Width;
            public int Height;
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
            SetPRD(0.08f, 0.08f, 0.12f);
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            SetPRD(0.75f, 0.55f, 0.35f);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private static void SetPRD(float r, float g, float b)
        {
            OptixPayloadInterop.SetFloat(0, r);
            OptixPayloadInterop.SetFloat(1, g);
            OptixPayloadInterop.SetFloat(2, b);
        }

        private const int Width = 1024;
        private const int Height = 768;
        private const int TileWidth = Width / 2;
        private const int TileHeight = Height;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        // One-time startup-only scratch buffer for the relocate-vs-relocated
        // pixel-identical check (a correctness comparison, not the display path -
        // the CPU round-trip here is fine since nothing is ever shown from it).
        private MemoryBuffer1D<uint, Stride1D.Dense> checkBuffer;
        private uint[] checkPixels;

        // Zero-copy CUDA-GL interop: both tiles' pipelines write directly into this
        // single full-window buffer every frame (see LaunchParams.XOffset/Stride) -
        // no host round-trip anywhere in the live render loop.
        private CudaGlInteropDisplayBuffer interopBuffer;

        private BuiltAccelStructure activeAccel;
        private LaunchParams launchParams;

        private RayTracingPipeline<ClusterLaunchParams> clusterPipeline;
        private ClusterLaunchParams clusterLaunchParams;
        private MemoryBuffer1D<Vector3, Stride1D.Dense> clusterVertexBuffer;
        private MemoryBuffer1D<uint, Stride1D.Dense> clusterIndexBuffer;
        private MemoryBuffer1D<byte, Stride1D.Dense> clasOutput;
        private MemoryBuffer1D<ulong, Stride1D.Dense> clasHandles;
        private MemoryBuffer1D<byte, Stride1D.Dense> gasOutput;
        private BuiltAccelStructure clusterIas;
        private bool clusterSupported;

        private FullscreenQuad quad;

        // Mouse-drag orbit camera state - left-drag-to-orbit, using OpenTK's polled
        // MouseState. cameraYaw/cameraPitch are shared across tiles; each tile's own
        // center/distance live near its scene setup (meshCenter/ClusterCenter etc.).
        private float cameraYaw;
        private float cameraPitch;
        private float cameraFocal;
        private bool wasLeftMouseDown;
        private float lastMouseX;
        private float lastMouseY;
        private const float MouseSensitivity = 0.3f;

        // Kept as a field, not a ctor local - the GC has no visibility into the native
        // function pointer OptiX holds onto for the log callback's lifetime.
        private OptixLogCallback logCallback;

        private int frameCount;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample15: acceleration structure relocation + emitted AABBs");
            Console.WriteLine("Initializing CUDA + OptiX (validation mode ALL)...");

            context = ILGPU.Context.Create(b => b.Cuda().EnableAlgorithms());
            accelerator = context.CreateCudaAccelerator(0);

            // Validation mode ALL + a log callback surfaces OptiX's own descriptive
            // error/warning text instead of a bare OptixResult code with no message -
            // e.g. relocating into an uninitialized target buffer surfaces as a
            // driver-side message rather than a bare "unspecified launch failure"
            // CUDA exception.
            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            rayTracer = OptixRayTracer.Create(accelerator, new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.All,
            });

            Console.WriteLine("Compiling pipeline (raygen module via task-based compilation)...");
            var compileStopwatch = System.Diagnostics.Stopwatch.StartNew();
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .CompileRaygenWithTasks()
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>(ClosestHitRadiance, AnyHitRadiance))
                .MaxTraceDepth(2));
            compileStopwatch.Stop();
            pipeline.SetHitRecords<byte>(new byte[] { 0 });
            Console.WriteLine(
                $"Raygen module compiled via {pipeline.RaygenCompileTaskCount} OptixTask(s) " +
                $"in {compileStopwatch.Elapsed.TotalMilliseconds:0.0}ms (fanned across the .NET thread pool).");

            Console.WriteLine("Loading mesh (cow.obj)...");
            var (vertices, indices) = Model.LoadPositionsOnly("models/meshes/cow.obj");
            Console.WriteLine($"Loaded {vertices.Length} vertices, {indices.Length} triangles.");

            using var vertexBuffer = accelerator.Allocate1D(vertices);
            using var indexBuffer = accelerator.Allocate1D(indices);

            Console.WriteLine("Building GAS...");
            var accelBuilder = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(vertexBuffer, indexBuffer)
                .AllowCompaction()
                .EmitAabbs();

            var originalAccel = accelBuilder.Build();

            var (center, radius) = ComputeBounds(vertices);
            var aabb = originalAccel.Aabbs?[0] ?? default;
            Console.WriteLine(
                $"Emitted AABB: min=({aabb.MinX:0.###}, {aabb.MinY:0.###}, {aabb.MinZ:0.###}) " +
                $"max=({aabb.MaxX:0.###}, {aabb.MaxY:0.###}, {aabb.MaxZ:0.###})");
            bool aabbLooksRight =
                XMath.Abs(aabb.MinX - center.X) <= radius + 0.01f &&
                XMath.Abs(aabb.MaxX - center.X) <= radius + 0.01f &&
                (aabb.MaxX - aabb.MinX) > 0.01f;
            Console.WriteLine(aabbLooksRight
                ? "AABB matches the mesh's known bounds: PASS"
                : "AABB does NOT match the mesh's known bounds: FAIL");

            SetupCamera(center, radius);

            checkBuffer = accelerator.Allocate1D<uint>(TileWidth * TileHeight);
            checkPixels = new uint[TileWidth * TileHeight];

            Console.WriteLine("Rendering from the original acceleration structure...");
            var originalPixels = RenderToHost(originalAccel.TraversableHandle);

            Console.WriteLine("Relocating the acceleration structure...");
            var relocatedAccel = originalAccel.Relocate(rayTracer.DeviceContext, accelerator);
            Console.WriteLine($"Relocated: {originalAccel.TraversableHandle} -> {relocatedAccel.TraversableHandle}");

            Console.WriteLine("Rendering from the relocated acceleration structure...");
            var relocatedPixels = RenderToHost(relocatedAccel.TraversableHandle);

            bool identical = originalPixels.AsSpan().SequenceEqual(relocatedPixels);
            Console.WriteLine(identical
                ? "Original vs. relocated render: PASS (pixel-identical)"
                : "Original vs. relocated render: FAIL (pixels differ)");

            // The relocated structure is what drives the live render loop below - keeps
            // exercising it every frame instead of just the one-shot comparison above.
            originalAccel.Dispose();
            activeAccel = relocatedAccel;
            launchParams.Traversable = unchecked((ulong)activeAccel.TraversableHandle.ToInt64());

            Console.WriteLine("Device properties:");
            foreach (OptixDeviceProperty property in Enum.GetValues<OptixDeviceProperty>())
            {
                var value = rayTracer.DeviceContext.GetProperty(property);
                Console.WriteLine($"  {property} = {value}");
            }

            // ------------------------------------------------------------------
            // Disk cache control (optixDeviceContextSetCacheEnabled/GetCache*).
            // ------------------------------------------------------------------
            rayTracer.DeviceContext.SetCacheEnabled(true);
            bool cacheEnabled = rayTracer.DeviceContext.GetCacheEnabled();
            string cacheLocation = rayTracer.DeviceContext.GetCacheLocation();
            rayTracer.DeviceContext.SetCacheDatabaseSizes(1UL << 27, 1UL << 29);
            var (lowWaterMark, highWaterMark) = rayTracer.DeviceContext.GetCacheDatabaseSizes();
            Console.WriteLine($"Disk cache: enabled={cacheEnabled}, location=\"{cacheLocation}\", " +
                $"gc={lowWaterMark}/{highWaterMark}");
            Console.WriteLine(cacheEnabled
                ? "Disk cache enable round-trip: PASS"
                : "Disk cache enable round-trip: FAIL");
            Console.WriteLine(lowWaterMark == 1UL << 27 && highWaterMark == 1UL << 29
                ? "Disk cache watermark round-trip: PASS"
                : "Disk cache watermark round-trip: FAIL");

            uint clusterAccelSupport = rayTracer.DeviceContext.GetProperty(OptixDeviceProperty.ClusterAccel);
            clusterSupported = clusterAccelSupport != 0;
            interopBuffer = new CudaGlInteropDisplayBuffer(Width, Height, accelerator);
            launchParams.XOffset = 0;
            launchParams.Stride = Width;

            if (clusterSupported)
                BuildClusterScene();
            else
                Console.WriteLine("Cluster accels: UNSUPPORTED on this device/driver - the right tile stays blank.");

            // Re-apply now that clusterSupported/clusterLaunchParams are settled -
            // the first ApplyCameraOrbit() call (inside SetupCamera, above) ran before
            // the cluster tile existed, so its camera fields are still unset.
            ApplyCameraOrbit();

            quad = new FullscreenQuad();

            Console.WriteLine(!aabbLooksRight || !identical
                ? "Startup checks FAILED - see above. Rendering live anyway."
                : "Startup checks passed. Rendering live from the relocated GAS.");
            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit (orbits both tiles together). " +
                "Left tile = relocated GAS, right tile = cluster accels" +
                (clusterSupported ? "." : " (unsupported on this driver, blank).") +
                " Esc to quit.");
        }

        private unsafe void BuildClusterScene()
        {
            var deviceContext = rayTracer.DeviceContext;
            Console.WriteLine("Cluster accels: building 4 CLASes (2x2 quad grid), then a GAS over them...");

            // 4 quads, each its own cluster: 4 vertices + 2 triangles per quad,
            // sharing one vertex array (per-cluster VertexBuffer pointers offset
            // into it) and one local index pattern.
            const int ClusterCount = 4;
            var vertices = new Vector3[ClusterCount * 4];
            for (int c = 0; c < ClusterCount; c++)
            {
                float x0 = (c % 2) * 2.2f;
                float y0 = (c / 2) * 2.2f;
                vertices[c * 4 + 0] = new Vector3(x0, y0, 0f);
                vertices[c * 4 + 1] = new Vector3(x0 + 2f, y0, 0f);
                vertices[c * 4 + 2] = new Vector3(x0 + 2f, y0 + 2f, 0f);
                vertices[c * 4 + 3] = new Vector3(x0, y0 + 2f, 0f);
            }
            var localIndices = new uint[] { 0, 1, 2, 0, 2, 3 };

            clusterVertexBuffer = accelerator.Allocate1D(vertices);
            clusterIndexBuffer = accelerator.Allocate1D(localIndices);

            try
            {
                // ----- CLAS from triangles (implicit destinations) -----
                var clasInput = new OptixClusterAccelBuildInput
                {
                    Type = OptixClusterAccelBuildType.ClustersFromTriangles,
                };
                clasInput.Triangles.Flags = OptixClusterAccelBuildFlags.None;
                clasInput.Triangles.MaxArgCount = ClusterCount;
                clasInput.Triangles.VertexFormat = OptixVertexFormat.Float3;
                clasInput.Triangles.MaxSbtIndexValue = 0;
                clasInput.Triangles.MaxUniqueSbtIndexCountPerArg = 1;
                clasInput.Triangles.MaxTriangleCountPerArg = 2;
                clasInput.Triangles.MaxVertexCountPerArg = 4;

                var clasSizes = deviceContext.ClusterAccelComputeMemoryUsage(
                    OptixClusterAccelBuildMode.ImplicitDestinations, clasInput);
                Console.WriteLine($"  CLAS sizes: output={clasSizes.OutputSizeInBytes}, temp={clasSizes.TempSizeInBytes}");

                clasOutput = accelerator.Allocate1D<byte>(
                    (long)Math.Max(1UL, clasSizes.OutputSizeInBytes));
                using var clasTemp = accelerator.Allocate1D<byte>(
                    (long)Math.Max(1UL, clasSizes.TempSizeInBytes));
                clasHandles = accelerator.Allocate1D<ulong>(ClusterCount);
                clasHandles.MemSetToZero();

                ulong vertexBase = unchecked((ulong)clusterVertexBuffer.NativePtr.ToInt64());
                var clasArgs = new OptixClusterAccelBuildInputTrianglesArgs[ClusterCount];
                for (int c = 0; c < ClusterCount; c++)
                {
                    clasArgs[c] = new OptixClusterAccelBuildInputTrianglesArgs
                    {
                        ClusterId = BaseClusterId + (uint)c,
                        ClusterFlags = (uint)OptixClusterAccelClusterFlags.None,
                        PackedCounts = OptixClusterAccelBuildInputTrianglesArgs.PackCounts(
                            triangleCount: 2,
                            vertexCount: 4,
                            indexFormat: OptixClusterAccelIndicesFormat.Bits32),
                        BasePrimitiveInfo = OptixClusterAccelPrimitiveInfo.Pack(0),
                        IndexBuffer = unchecked((ulong)clusterIndexBuffer.NativePtr.ToInt64()),
                        VertexBuffer = vertexBase + (ulong)(c * 4 * 3 * sizeof(float)),
                    };
                }
                using var clasArgsBuffer = accelerator.Allocate1D(clasArgs);

                var clasModeDesc = new OptixClusterAccelBuildModeDesc
                {
                    Mode = OptixClusterAccelBuildMode.ImplicitDestinations,
                };
                clasModeDesc.ImplicitDest.OutputBuffer =
                    unchecked((ulong)clasOutput.NativePtr.ToInt64());
                clasModeDesc.ImplicitDest.OutputBufferSizeInBytes = clasSizes.OutputSizeInBytes;
                clasModeDesc.ImplicitDest.TempBuffer =
                    unchecked((ulong)clasTemp.NativePtr.ToInt64());
                clasModeDesc.ImplicitDest.TempBufferSizeInBytes = clasSizes.TempSizeInBytes;
                clasModeDesc.ImplicitDest.OutputHandlesBuffer =
                    unchecked((ulong)clasHandles.NativePtr.ToInt64());
                clasModeDesc.ImplicitDest.OutputHandlesStrideInBytes = 0;

                deviceContext.ClusterAccelBuild(
                    accelerator.DefaultStream,
                    clasModeDesc,
                    clasInput,
                    unchecked((ulong)clasArgsBuffer.NativePtr.ToInt64()));
                accelerator.Synchronize();
                var handles = clasHandles.GetAsArray1D();
                Console.WriteLine(Array.TrueForAll(handles, h => h != 0)
                    ? $"  CLAS build returned handles: PASS (first=0x{handles[0]:X})"
                    : "  CLAS build returned handles: FAIL");

                // ----- GAS from clusters (implicit destinations) -----
                var gasInput = new OptixClusterAccelBuildInput
                {
                    Type = OptixClusterAccelBuildType.GasesFromClusters,
                };
                gasInput.Clusters.Flags = OptixClusterAccelBuildFlags.None;
                gasInput.Clusters.MaxArgCount = 1;
                gasInput.Clusters.MaxTotalClusterCount = ClusterCount;
                gasInput.Clusters.MaxClusterCountPerArg = ClusterCount;

                var gasSizes = deviceContext.ClusterAccelComputeMemoryUsage(
                    OptixClusterAccelBuildMode.ImplicitDestinations, gasInput);
                Console.WriteLine($"  GAS sizes: output={gasSizes.OutputSizeInBytes}, temp={gasSizes.TempSizeInBytes}");

                gasOutput = accelerator.Allocate1D<byte>(
                    (long)Math.Max(1UL, gasSizes.OutputSizeInBytes));
                using var gasTemp = accelerator.Allocate1D<byte>(
                    (long)Math.Max(1UL, gasSizes.TempSizeInBytes));
                using var gasHandlesBuf = accelerator.Allocate1D<ulong>(1);
                gasHandlesBuf.MemSetToZero();

                var gasArgs = new[]
                {
                    new OptixClusterAccelBuildInputClustersArgs
                    {
                        ClusterHandlesCount = ClusterCount,
                        ClusterHandlesBufferStrideInBytes = 0,
                        ClusterHandlesBuffer =
                            unchecked((ulong)clasHandles.NativePtr.ToInt64()),
                    },
                };
                using var gasArgsBuffer = accelerator.Allocate1D(gasArgs);

                var gasModeDesc = new OptixClusterAccelBuildModeDesc
                {
                    Mode = OptixClusterAccelBuildMode.ImplicitDestinations,
                };
                gasModeDesc.ImplicitDest.OutputBuffer =
                    unchecked((ulong)gasOutput.NativePtr.ToInt64());
                gasModeDesc.ImplicitDest.OutputBufferSizeInBytes = gasSizes.OutputSizeInBytes;
                gasModeDesc.ImplicitDest.TempBuffer =
                    unchecked((ulong)gasTemp.NativePtr.ToInt64());
                gasModeDesc.ImplicitDest.TempBufferSizeInBytes = gasSizes.TempSizeInBytes;
                gasModeDesc.ImplicitDest.OutputHandlesBuffer =
                    unchecked((ulong)gasHandlesBuf.NativePtr.ToInt64());
                gasModeDesc.ImplicitDest.OutputHandlesStrideInBytes = 0;

                deviceContext.ClusterAccelBuild(
                    accelerator.DefaultStream,
                    gasModeDesc,
                    gasInput,
                    unchecked((ulong)gasArgsBuffer.NativePtr.ToInt64()));
                accelerator.Synchronize();
                ulong gasHandle = gasHandlesBuf.GetAsArray1D()[0];
                Console.WriteLine(gasHandle != 0
                    ? $"  cluster GAS build returned a traversable: PASS (0x{gasHandle:X})"
                    : "  cluster GAS build returned a traversable: FAIL");

                // Not .CompileRaygenWithTasks() here - task-based compilation is
                // already demonstrated by the left tile's pipeline; combining it
                // with .AllowClusteredGeometry() hits OPTIX_ERROR_INVALID_VALUE on
                // this driver, so the cluster pipeline just compiles normally.
                Console.WriteLine("  Compiling clustered-geometry pipeline...");
                clusterPipeline = rayTracer.CreatePipeline<ClusterLaunchParams>(b => b
                    .Raygen(ClusterRaygen)
                    .RayType("radiance", r => r
                        .Payload<ClusterRadiancePayload>()
                        .Miss(ClusterMiss)
                        .HitGroup<byte>(ClusterClosestHit))
                    .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                    .AllowClusteredGeometry()
                    .MaxTraceDepth(2));
                clusterPipeline.SetHitRecords<byte>(new byte[] { 0 });

                clusterIas = new OptixAccelBuilder()
                    .WithDeviceContext(deviceContext)
                    .WithAccelerator(accelerator)
                    .BuildInstanceAccelFromHandles(
                        new[] { new IntPtr(unchecked((long)gasHandle)) },
                        new uint[] { 0 });

                clusterLaunchParams.Traversable = unchecked((ulong)clusterIas.TraversableHandle.ToInt64());
                clusterLaunchParams.XOffset = TileWidth;
                clusterLaunchParams.Stride = Width;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Cluster accel setup FAILED: {e.Message}");
                clusterSupported = false;
            }
        }

        // Only cameraYaw/cameraPitch are shared (one mouse drag orbits every tile in
        // sync); each tile keeps its own fixed scene center/orbit distance and
        // recomputes its own basis from the shared yaw/pitch every frame.
        private Vector3 meshCenter;
        private float meshOrbitDistance;
        private static readonly Vector3 ClusterCenter = new Vector3(2.1f, 2.1f, 0f);
        private const float ClusterOrbitDistance = 7f;

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
            meshCenter = center;
            meshOrbitDistance = radius * 1.6f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.9f, 0.5f, 1.4f));
            cameraYaw = MathF.Atan2(initialDir.X, initialDir.Z) * 180f / MathF.PI;
            cameraPitch = MathF.Asin(Math.Clamp(initialDir.Y, -1f, 1f)) * 180f / MathF.PI;

            launchParams = new LaunchParams { Width = TileWidth, Height = TileHeight };
            ApplyCameraOrbit();
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

        // Recomputes both tiles' camera basis from the shared yaw/pitch - called once
        // at startup and again on every mouse-drag frame. Both tiles always render, so
        // both always need up-to-date camera params, not just the "active" one.
        private void ApplyCameraOrbit()
        {
            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, meshCenter, meshOrbitDistance);
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

            if (!clusterSupported)
                return;

            var (cOrigin, cForward, cRight, cUp) = OrbitBasis(cameraYaw, cameraPitch, ClusterCenter, ClusterOrbitDistance);
            clusterLaunchParams.OriginX = cOrigin.X;
            clusterLaunchParams.OriginY = cOrigin.Y;
            clusterLaunchParams.OriginZ = cOrigin.Z;
            clusterLaunchParams.ForwardX = cForward.X;
            clusterLaunchParams.ForwardY = cForward.Y;
            clusterLaunchParams.ForwardZ = cForward.Z;
            clusterLaunchParams.RightX = cRight.X;
            clusterLaunchParams.RightY = cRight.Y;
            clusterLaunchParams.RightZ = cRight.Z;
            clusterLaunchParams.UpX = cUp.X;
            clusterLaunchParams.UpY = cUp.Y;
            clusterLaunchParams.UpZ = cUp.Z;
            clusterLaunchParams.Focal = cameraFocal;
        }

        private void UpdateMouseOrbit()
        {
            bool leftDown = MouseState.IsButtonDown(MouseButton.Left);
            float x = MouseState.X;
            float y = MouseState.Y;

            if (leftDown && !wasLeftMouseDown)
            {
                // Just pressed - reset the delta baseline so the view doesn't jump on
                // the first drag frame.
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

        // Startup-only correctness check (compares original vs. relocated-GAS
        // renders byte-for-byte) - runs before the interop buffer exists, so it
        // uses its own plain device buffer + one CPU readback. Not part of the live
        // display path.
        private byte[] RenderToHost(IntPtr traversable)
        {
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            launchParams.XOffset = 0;
            launchParams.Stride = TileWidth;
            launchParams.Traversable = unchecked((ulong)traversable.ToInt64());

            pipeline.Launch(launchParams, TileWidth, TileHeight);
            accelerator.Synchronize();

            checkBuffer.CopyToCPU(checkPixels);
            return MemoryMarshal.AsBytes(checkPixels.AsSpan()).ToArray();
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
            interopBuffer.GetCudaArrayView(); // sets interopBuffer.NativePtr to the mapped device pointer
            var fullWindowView = new OptixDeviceView<uint>((uint*)interopBuffer.NativePtr, Width * (long)Height);

            launchParams.ColorBuffer = fullWindowView;
            pipeline.Launch(launchParams, TileWidth, TileHeight);

            if (clusterSupported)
            {
                clusterLaunchParams.ColorBuffer = fullWindowView;
                clusterPipeline.Launch(clusterLaunchParams, TileWidth, TileHeight);
            }

            accelerator.Synchronize();
            interopBuffer.UnmapCuda(stream);
            interopBuffer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(interopBuffer.GlTextureHandle);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample15] frame {frameCount}");
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

            clusterIas?.Dispose();
            gasOutput?.Dispose();
            clasHandles?.Dispose();
            clasOutput?.Dispose();
            clusterIndexBuffer?.Dispose();
            clusterVertexBuffer?.Dispose();
            clusterPipeline?.Dispose();

            activeAccel?.Dispose();
            checkBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
