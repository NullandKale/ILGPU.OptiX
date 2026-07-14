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

namespace Sample22
{
    /// <summary>
    /// Exercises the OptiX denoiser's temporal
    /// model kind and its internal guide layers (OptixDenoiserBuilder now allocates +
    /// ping-pongs the two opaque internal-guide-layer buffers automatically via
    /// BuiltDenoiser.PrepareTemporalGuideLayer - previously only the basic Hdr/Ldr
    /// path was ever wired up by any sample).
    ///
    /// Uses a real, non-deterministic noise source (not just a uniform per-pixel
    /// jitter, which is too structureless to visibly show a denoiser doing anything -
    /// per user feedback while validating this sample): a single Monte Carlo shadow
    /// ray per pixel per frame, jittered toward a random point on a small disk-shaped
    /// area light, using the hit-distance-in-payload trick from Sample18 to
    /// reconstruct the primary ray's hit point. This produces classic 1-sample soft-
    /// shadow speckle that flickers frame to frame in plain Hdr mode - the kind of
    /// signal a denoiser (and temporal accumulation specifically) is actually built to
    /// clean up. Press 'T' to toggle between Hdr (basic, no temporal history) and
    /// TemporalAov (uses the previous frame's denoised output + internal guide layers
    /// to stabilize the result) and compare the shadow noise in each. The flow
    /// (motion vector) guide layer is supplied as all-zero - a documented
    /// simplification (this device API layer has no scene motion-vector/reprojection
    /// support yet), valid for a static camera and a reasonable approximation for slow
    /// orbiting.
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

        // radiance: r,g,b + hit distance (bit-cast float, -1 = miss), same trick as Sample18.
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

        private const int Width = 800;
        private const int Height = 600;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<Vector4, Stride1D.Dense> noisyColorBuffer;
        private MemoryBuffer1D<Vector4, Stride1D.Dense> denoisedColorBuffer;
        private MemoryBuffer1D<Vector4, Stride1D.Dense> flowBuffer;
        private Vector4[] denoisedHost;
        private uint[] pixelHost;

        private BuiltAccelStructure accel;
        private BuiltDenoiser denoiser;
        private bool temporalEnabled;
        private bool firstTemporalFrame;
        private LaunchParams launchParams;

        private int glTexture;
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

        private bool tWasDown;
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

            Console.WriteLine("Sample22: denoiser temporal model kind + internal guide layers");
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

            noisyColorBuffer = accelerator.Allocate1D<Vector4>(Width * Height);
            denoisedColorBuffer = accelerator.Allocate1D<Vector4>(Width * Height);
            flowBuffer = accelerator.Allocate1D<Vector4>(Width * Height);
            flowBuffer.MemSetToZero(); // documented simplification - see class doc comment
            denoisedHost = new Vector4[Width * Height];
            pixelHost = new uint[Width * Height];

            launchParams.Width = Width;
            launchParams.Height = Height;
            launchParams.NoisyColorBuffer = OptixDeviceView<Vector4>.From(noisyColorBuffer);
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            BuildDenoiser(OptixDenoiserModelKind.Hdr);

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            Console.WriteLine("Denoiser build (Hdr): PASS.");
            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, T toggles Hdr/TemporalAov denoiser, Esc to quit.");
        }

        private void BuildDenoiser(OptixDenoiserModelKind modelKind)
        {
            denoiser?.Dispose();
            denoiser = new OptixDenoiserBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .WithImageDimensions((uint)Width, (uint)Height)
                .WithDenoiserOptions(new OptixDenoiserOptions { GuideAlbedo = 0, GuideNormal = 0 })
                .WithModelKind(modelKind)
                .Build();
            firstTemporalFrame = true;
            Title = $"Sample22 - Denoiser [{(temporalEnabled ? "TemporalAov" : "Hdr")}]";
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

            bool tDown = KeyboardState.IsKeyDown(Keys.T);
            if (tDown && !tWasDown)
            {
                temporalEnabled = !temporalEnabled;
                BuildDenoiser(temporalEnabled ? OptixDenoiserModelKind.TemporalAov : OptixDenoiserModelKind.Hdr);
                Console.WriteLine($"[Denoiser] {(temporalEnabled ? "TemporalAov" : "Hdr")}");
            }
            tWasDown = tDown;

            UpdateMouseOrbit();
        }

        private OptixImage2D MakeImage(MemoryBuffer1D<Vector4, Stride1D.Dense> buffer) => new OptixImage2D
        {
            Data = unchecked((ulong)buffer.NativePtr.ToInt64()),
            Width = (uint)Width,
            Height = (uint)Height,
            RowStrideInBytes = (uint)(Width * 16), // sizeof(Vector4)
            PixelStrideInBytes = 16,
            Format = OptixPixelFormat.Float4,
        };

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            frameSeed++;
            launchParams.FrameSeed = frameSeed;
            pipeline.Launch(launchParams, Width, Height);

            var stream = accelerator.DefaultStream as CudaStream;
            var noisyImage = MakeImage(noisyColorBuffer);
            var outputImage = MakeImage(denoisedColorBuffer);

            var guideLayer = default(OptixDenoiserGuideLayer);
            if (temporalEnabled)
            {
                var flowImage = MakeImage(flowBuffer);
                guideLayer = denoiser.PrepareTemporalGuideLayer(guideLayer, flowImage);
            }

            var layer = new OptixDenoiserLayer
            {
                Input = noisyImage,
                Output = outputImage,
                PreviousOutput = temporalEnabled ? outputImage : default,
                Type = OptixDenoiserAOVType.Beauty,
            };

            var parameters = new OptixDenoiserParams
            {
                HdrIntensity = 0,
                BlendFactor = 0f,
                HdrAverageColor = 0,
                TemporalModeUsePreviousLayers = (temporalEnabled && !firstTemporalFrame) ? 1u : 0u,
            };
            firstTemporalFrame = false;

            denoiser.Denoiser.Invoke(
                stream.StreamPtr,
                parameters,
                denoiser.StateBuffer.NativePtr,
                (ulong)denoiser.StateBuffer.LengthInBytes,
                guideLayer,
                new[] { layer },
                denoiser.ScratchBuffer.NativePtr,
                (ulong)denoiser.ScratchBuffer.LengthInBytes);

            accelerator.Synchronize();
            denoisedColorBuffer.CopyToCPU(denoisedHost);

            for (int i = 0; i < denoisedHost.Length; i++)
            {
                var c = denoisedHost[i];
                uint r = (uint)(XMath.Clamp(c.X, 0f, 1f) * 255f);
                uint g = (uint)(XMath.Clamp(c.Y, 0f, 1f) * 255f);
                uint b = (uint)(XMath.Clamp(c.Z, 0f, 1f) * 255f);
                pixelHost[i] = 0xff000000u | (b << 16) | (g << 8) | r;
            }

            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, Width, Height,
                PixelFormat.Rgba, PixelType.UnsignedByte, pixelHost);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(glTexture);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample22] frame {frameCount} ({(temporalEnabled ? "TemporalAov" : "Hdr")})");
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }

        protected override void OnUnload()
        {
            quad?.Dispose();
            if (glTexture != 0)
                GL.DeleteTexture(glTexture);

            denoiser?.Dispose();
            accel?.Dispose();
            noisyColorBuffer?.Dispose();
            denoisedColorBuffer?.Dispose();
            flowBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
