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
using ILGPU.OptiX.Denoising;
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

namespace Sample19
{
    /// <summary>
    /// Denoiser API surface sample. Covers the Hdr and TemporalAov model kinds side
    /// by side - left tile Hdr (no temporal history), right tile TemporalAov (uses
    /// the previous frame's denoised output + internal guide layers, ping-ponged via
    /// BuiltDenoiser.PrepareTemporalGuideLayer) - both running every frame against
    /// the same noisy input, plus a startup-only check of the tiled denoiser
    /// (OptixDenoiserTiling) over a synthetic noisy gradient.
    ///
    /// Uses a real, non-deterministic noise source (not just a uniform per-pixel
    /// jitter, which is too structureless to visibly show a denoiser doing anything):
    /// a single Monte Carlo shadow ray per pixel per frame, jittered toward a random
    /// point on a small disk-shaped area light, using a hit-distance-in-payload trick
    /// to reconstruct the primary ray's hit point. This produces classic 1-sample
    /// soft-shadow speckle that flickers frame to frame - the kind of signal a
    /// denoiser (and temporal accumulation specifically) is actually built to clean
    /// up; the right tile should visibly stabilize faster than the left as frames
    /// accumulate. The flow (motion vector) guide layer is supplied as all-zero - a
    /// documented simplification (this device API layer has no scene
    /// motion-vector/reprojection support yet), valid for a static camera and a
    /// reasonable approximation for slow orbiting.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private struct LaunchParams
        {
            public OptixDeviceView<Vector4> NoisyColorBuffer;
            public int Width;
            public int Height;
            public ulong Traversable;
            public uint FrameSeed;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
            public float LightX, LightY, LightZ;
            public float LightRadius;
        }

        // radiance: r,g,b + hit distance (bit-cast float, -1 = miss).
        private struct RadiancePayload
        {
            public uint P0, P1, P2, P3;
        }

        // shadow: 1 = occluded, 0 = light sample visible.
        private struct ShadowPayload
        {
            public uint P0;
        }

        // Simple integer hash (Wang hash) for cheap per-pixel-per-frame randomness.
        private static uint Hash(uint x)
        {
            x = (x ^ 61u) ^ (x >> 16);
            x *= 9u;
            x ^= x >> 4;
            x *= 0x27d4eb2du;
            x ^= x >> 15;
            return x;
        }

        private static float Rand01(uint seed) => (Hash(seed) & 0xFFFFu) / 65535f;

        private static void RaygenRenderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            uint pixelIndex = (uint)(ix + iy * launchParams.Width);
            uint seed = pixelIndex * 9781u + launchParams.FrameSeed * 6271u;

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

            uint p0 = 0, p1 = 0, p2 = 0, p3 = 0;
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
                2,
                0,
                ref p0,
                ref p1,
                ref p2,
                ref p3);

            float r = Interop.IntAsFloat(p0);
            float g = Interop.IntAsFloat(p1);
            float b = Interop.IntAsFloat(p2);
            float hitDistance = Interop.IntAsFloat(p3);

            // One Monte Carlo shadow sample per pixel per frame, jittered toward a
            // random point on a small disk area light - this is what actually
            // produces real (geometry-correlated, spatially structured) 1-sample
            // noise for the denoiser to clean up, unlike a flat per-pixel jitter.
            float shadowFactor = 1f;
            if (hitDistance > 0f)
            {
                float hitX = launchParams.OriginX + dx * hitDistance;
                float hitY = launchParams.OriginY + dy * hitDistance;
                float hitZ = launchParams.OriginZ + dz * hitDistance;

                float a1 = Rand01(seed);
                float a2 = Rand01(seed ^ 0x9e3779b9u);
                float radius = launchParams.LightRadius * XMath.Sqrt(a1);
                float angle = a2 * 2f * XMath.PI;
                float sampleX = launchParams.LightX + radius * XMath.Cos(angle);
                float sampleY = launchParams.LightY;
                float sampleZ = launchParams.LightZ + radius * XMath.Sin(angle);

                float sdx = sampleX - hitX;
                float sdy = sampleY - hitY;
                float sdz = sampleZ - hitZ;
                float sLen = XMath.Sqrt(sdx * sdx + sdy * sdy + sdz * sdz);
                float sInvLen = 1f / sLen;
                sdx *= sInvLen;
                sdy *= sInvLen;
                sdz *= sInvLen;

                float shadowOriginX = hitX + dx * 1e-3f;
                float shadowOriginY = hitY + dy * 1e-3f;
                float shadowOriginZ = hitZ + dz * 1e-3f;

                uint occluded = 0;
                OptixTrace.Trace(
                    launchParams.Traversable,
                    (shadowOriginX, shadowOriginY, shadowOriginZ),
                    (sdx, sdy, sdz),
                    1e-3f,
                    sLen - 1e-2f,
                    0f,
                    0xff,
                    OptixRayFlags.TerminateOnFirstHit,
                    1,
                    2,
                    1,
                    ref occluded);

                shadowFactor = occluded == 1u ? 0.15f : 1.0f;
            }

            long pixel = ix + (long)iy * launchParams.Width;
            launchParams.NoisyColorBuffer[pixel] = new Vector4(
                XMath.Clamp(r * shadowFactor, 0f, 1f),
                XMath.Clamp(g * shadowFactor, 0f, 1f),
                XMath.Clamp(b * shadowFactor, 0f, 1f),
                1f);
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.08f);
            OptixPayloadInterop.SetFloat(1, 0.08f);
            OptixPayloadInterop.SetFloat(2, 0.12f);
            OptixPayloadInterop.SetFloat(3, -1f);
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.75f);
            OptixPayloadInterop.SetFloat(1, 0.55f);
            OptixPayloadInterop.SetFloat(2, 0.3f);
            OptixPayloadInterop.SetFloat(3, OptixGetRayTmax.Value);
        }

        private static void MissShadow(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetUint(0, 0);
        }

        private static void ClosestHitShadow(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetUint(0, 1);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private const int TileWidth = 800;
        private const int TileHeight = 600;
        private const int Width = TileWidth * 2;
        private const int Height = TileHeight;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<Vector4, Stride1D.Dense> noisyColorBuffer;
        private MemoryBuffer1D<Vector4, Stride1D.Dense> denoisedHdrBuffer;
        private MemoryBuffer1D<Vector4, Stride1D.Dense> denoisedTemporalBuffer;
        private MemoryBuffer1D<Vector4, Stride1D.Dense> flowBuffer;

        // Packs a denoised tile straight into the shared interop buffer on the GPU
        // (see Device/PackKernel.cs) - no CPU round-trip for the denoised image.
        private Action<Index1D, int, int, int, ArrayView<Vector4>, ArrayView<byte>> packKernel;

        private BuiltAccelStructure accel;
        private BuiltDenoiser denoiserHdr;
        private BuiltDenoiser denoiserTemporal;
        private bool firstTemporalFrame = true;
        private LaunchParams launchParams;

        private CudaGlInteropDisplayBuffer interopBuffer;
        private FullscreenQuad quad;

        private OptixLogCallback logCallback;

        private Vector3 sceneCenter;
        private float orbitDistance;
        private float cameraYaw;
        private float cameraPitch;
        private float cameraFocal;
        private bool wasLeftMouseDown;
        private float lastMouseX;
        private float lastMouseY;
        private const float MouseSensitivity = 0.3f;

        private uint frameSeed;
        private int frameCount;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample19: denoiser temporal model kind + internal guide layers");
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

            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>(ClosestHitRadiance, AnyHitRadiance))
                .RayType("shadow", r => r
                    .Payload<ShadowPayload>()
                    .Miss(MissShadow)
                    .HitGroup<byte>(ClosestHitShadow))
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            Console.WriteLine("Loading mesh (cow.obj)...");
            var (vertices, indices) = Model.LoadPositionsOnly("models/meshes/cow.obj");

            using var vertexBuffer = accelerator.Allocate1D(vertices);
            using var indexBuffer = accelerator.Allocate1D(indices);

            accel = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(vertexBuffer, indexBuffer)
                .AllowCompaction()
                .Build();

            var (center, radius) = ComputeBounds(vertices);
            SetupCamera(center, radius);

            noisyColorBuffer = accelerator.Allocate1D<Vector4>(TileWidth * TileHeight);
            denoisedHdrBuffer = accelerator.Allocate1D<Vector4>(TileWidth * TileHeight);
            denoisedTemporalBuffer = accelerator.Allocate1D<Vector4>(TileWidth * TileHeight);
            flowBuffer = accelerator.Allocate1D<Vector4>(TileWidth * TileHeight);
            flowBuffer.MemSetToZero(); // documented simplification - see class doc comment
            packKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, int, int, int, ArrayView<Vector4>, ArrayView<byte>>(PackKernel.PackToDisplay);

            launchParams.Width = TileWidth;
            launchParams.Height = TileHeight;
            launchParams.NoisyColorBuffer = OptixDeviceView<Vector4>.From(noisyColorBuffer);
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            denoiserHdr = new OptixDenoiserBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .WithImageDimensions((uint)TileWidth, (uint)TileHeight)
                .WithDenoiserOptions(new OptixDenoiserOptions { GuideAlbedo = 0, GuideNormal = 0 })
                .WithModelKind(OptixDenoiserModelKind.Hdr)
                .Build();
            denoiserTemporal = new OptixDenoiserBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .WithImageDimensions((uint)TileWidth, (uint)TileHeight)
                .WithDenoiserOptions(new OptixDenoiserOptions { GuideAlbedo = 0, GuideNormal = 0 })
                .WithModelKind(OptixDenoiserModelKind.TemporalAov)
                .Build();
            RunTiledDenoiseCheck();

            interopBuffer = new CudaGlInteropDisplayBuffer(Width, Height, accelerator);
            quad = new FullscreenQuad();

            Console.WriteLine("Denoiser builds (Hdr + TemporalAov): PASS.");
            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit. Left tile = Hdr, right tile = TemporalAov. Esc to quit.");
        }

        private void RunTiledDenoiseCheck()
        {
            const uint tileSize = 128;
            const uint imageSize = 512;

            Console.WriteLine("Tiled denoising: creating denoiser sized for " +
                $"{tileSize}x{tileSize} tiles over a {imageSize}x{imageSize} image...");

            using var tiledDenoiser = rayTracer.DeviceContext.CreateDenoiser(
                OptixDenoiserModelKind.Hdr,
                new OptixDenoiserOptions { GuideAlbedo = 0, GuideNormal = 0 });

            var sizes = tiledDenoiser.ComputeMemoryResources(tileSize, tileSize);
            uint overlap = sizes.OverlapWindowSizeInPixels;

            uint setupWidth = tileSize + 2 * overlap;
            uint setupHeight = tileSize + 2 * overlap;
            using var stateBuffer = accelerator.Allocate1D<byte>((long)sizes.StateSizeInBytes);
            using var scratchBuffer = accelerator.Allocate1D<byte>(
                (long)sizes.WithOverlapScratchSizeInBytes);

            var stream = (CudaStream)accelerator.DefaultStream;
            tiledDenoiser.Setup(
                stream.StreamPtr,
                setupWidth,
                setupHeight,
                stateBuffer.NativePtr,
                (ulong)stateBuffer.LengthInBytes,
                scratchBuffer.NativePtr,
                (ulong)scratchBuffer.LengthInBytes);
            accelerator.Synchronize();

            // A noisy gradient as float4 input.
            var rng = new Random(1234);
            long pixelCount = imageSize * imageSize;
            var inputHost = new float[pixelCount * 4];
            for (uint y = 0; y < imageSize; y++)
            {
                for (uint x = 0; x < imageSize; x++)
                {
                    long i = (y * imageSize + x) * 4;
                    float noise = (float)rng.NextDouble() * 0.4f;
                    inputHost[i + 0] = (float)x / imageSize + noise;
                    inputHost[i + 1] = (float)y / imageSize + noise;
                    inputHost[i + 2] = 0.5f + noise;
                    inputHost[i + 3] = 1f;
                }
            }

            using var inputBuffer = accelerator.Allocate1D(inputHost);
            using var outputBuffer = accelerator.Allocate1D<float>(inputHost.Length);
            outputBuffer.MemSetToZero();

            OptixImage2D MakeTiledImage(IntPtr data) => new OptixImage2D
            {
                Data = unchecked((ulong)data.ToInt64()),
                Width = imageSize,
                Height = imageSize,
                RowStrideInBytes = imageSize * 4 * sizeof(float),
                PixelStrideInBytes = 4 * sizeof(float),
                Format = OptixPixelFormat.Float4,
            };

            var tiledLayer = new OptixDenoiserLayer
            {
                Input = MakeTiledImage(inputBuffer.NativePtr),
                Output = MakeTiledImage(outputBuffer.NativePtr),
            };

            tiledDenoiser.InvokeTiled(
                stream.StreamPtr,
                new OptixDenoiserParams(),
                stateBuffer.NativePtr,
                (ulong)stateBuffer.LengthInBytes,
                default,
                new[] { tiledLayer },
                scratchBuffer.NativePtr,
                (ulong)scratchBuffer.LengthInBytes,
                overlap,
                tileSize,
                tileSize);
            accelerator.Synchronize();

            var outputHost = outputBuffer.GetAsArray1D();
            int nonZero = 0;
            double difference = 0;
            for (int i = 0; i < outputHost.Length; i += 4)
            {
                if (outputHost[i] != 0f)
                    nonZero++;
                difference += Math.Abs(outputHost[i] - inputHost[i]);
            }
            Console.WriteLine(nonZero > pixelCount / 2
                ? $"Tiled denoise produced output: PASS ({nonZero} non-zero of {pixelCount} pixels)"
                : $"Tiled denoise produced output: FAIL ({nonZero} non-zero of {pixelCount} pixels)");
            Console.WriteLine(difference > 0
                ? $"Tiled denoise changed the noisy input: PASS (sum|out-in|={difference:F1})"
                : "Tiled denoise changed the noisy input: FAIL");
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
            sceneCenter = center;
            orbitDistance = radius * 1.6f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.9f, 0.5f, 1.4f));
            cameraYaw = MathF.Atan2(initialDir.X, initialDir.Z) * 180f / MathF.PI;
            cameraPitch = MathF.Asin(Math.Clamp(initialDir.Y, -1f, 1f)) * 180f / MathF.PI;

            // A small disk area light above and to the side of the mesh - sized
            // relative to the mesh so the penumbra (the actually-noisy region) is
            // clearly visible without being either a hard-edged point light or a
            // fully-diffuse blur.
            launchParams.LightX = center.X + radius * 1.2f;
            launchParams.LightY = center.Y + radius * 2f;
            launchParams.LightZ = center.Z + radius * 0.6f;
            launchParams.LightRadius = radius * 0.6f;

            ApplyCameraOrbit();
        }

        private void ApplyCameraOrbit()
        {
            float yawRad = cameraYaw * MathF.PI / 180f;
            float pitchRad = cameraPitch * MathF.PI / 180f;

            var dir = new Vector3(
                MathF.Cos(pitchRad) * MathF.Sin(yawRad),
                MathF.Sin(pitchRad),
                MathF.Cos(pitchRad) * MathF.Cos(yawRad));

            var origin = sceneCenter + dir * orbitDistance;
            var forward = Vector3.Normalize(sceneCenter - origin);
            var worldUp = new Vector3(0f, 1f, 0f);
            var right = Vector3.Normalize(Vector3.Cross(forward, worldUp));
            var up = Vector3.Cross(right, forward);

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

        private static OptixImage2D MakeImage(MemoryBuffer1D<Vector4, Stride1D.Dense> buffer) => new OptixImage2D
        {
            Data = unchecked((ulong)buffer.NativePtr.ToInt64()),
            Width = (uint)TileWidth,
            Height = (uint)TileHeight,
            RowStrideInBytes = (uint)(TileWidth * 16), // sizeof(Vector4)
            PixelStrideInBytes = 16,
            Format = OptixPixelFormat.Float4,
        };

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            frameSeed++;
            launchParams.FrameSeed = frameSeed;
            pipeline.Launch(launchParams, TileWidth, TileHeight);

            var stream = accelerator.DefaultStream as CudaStream;
            var noisyImage = MakeImage(noisyColorBuffer);

            // Left tile: Hdr, no temporal history.
            var hdrOutputImage = MakeImage(denoisedHdrBuffer);
            var hdrLayer = new OptixDenoiserLayer
            {
                Input = noisyImage,
                Output = hdrOutputImage,
                Type = OptixDenoiserAOVType.Beauty,
            };
            denoiserHdr.Denoiser.Invoke(
                stream.StreamPtr,
                new OptixDenoiserParams(),
                denoiserHdr.StateBuffer.NativePtr,
                (ulong)denoiserHdr.StateBuffer.LengthInBytes,
                default,
                new[] { hdrLayer },
                denoiserHdr.ScratchBuffer.NativePtr,
                (ulong)denoiserHdr.ScratchBuffer.LengthInBytes);

            // Right tile: TemporalAov, using the previous frame's denoised output +
            // internal guide layers (ping-ponged by PrepareTemporalGuideLayer).
            var flowImage = MakeImage(flowBuffer);
            var temporalOutputImage = MakeImage(denoisedTemporalBuffer);
            var temporalGuideLayer = denoiserTemporal.PrepareTemporalGuideLayer(default, flowImage);
            var temporalLayer = new OptixDenoiserLayer
            {
                Input = noisyImage,
                Output = temporalOutputImage,
                PreviousOutput = temporalOutputImage,
                Type = OptixDenoiserAOVType.Beauty,
            };
            var temporalParams = new OptixDenoiserParams
            {
                HdrIntensity = 0,
                BlendFactor = 0f,
                HdrAverageColor = 0,
                TemporalModeUsePreviousLayers = firstTemporalFrame ? 0u : 1u,
            };
            firstTemporalFrame = false;
            denoiserTemporal.Denoiser.Invoke(
                stream.StreamPtr,
                temporalParams,
                denoiserTemporal.StateBuffer.NativePtr,
                (ulong)denoiserTemporal.StateBuffer.LengthInBytes,
                temporalGuideLayer,
                new[] { temporalLayer },
                denoiserTemporal.ScratchBuffer.NativePtr,
                (ulong)denoiserTemporal.ScratchBuffer.LengthInBytes);

            interopBuffer.MapCuda(stream);
            var displayView = interopBuffer.GetCudaArrayView();

            packKernel(new Index1D(TileWidth * TileHeight), TileWidth, 0, Width,
                denoisedHdrBuffer.View, displayView);
            packKernel(new Index1D(TileWidth * TileHeight), TileWidth, TileWidth, Width,
                denoisedTemporalBuffer.View, displayView);

            accelerator.Synchronize();
            interopBuffer.UnmapCuda(stream);
            interopBuffer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(interopBuffer.GlTextureHandle);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample19] frame {frameCount}");
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

            denoiserHdr?.Dispose();
            denoiserTemporal?.Dispose();
            accel?.Dispose();
            noisyColorBuffer?.Dispose();
            denoisedHdrBuffer?.Dispose();
            denoisedTemporalBuffer?.Dispose();
            flowBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
