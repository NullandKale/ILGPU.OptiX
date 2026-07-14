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

namespace Sample23
{
    /// <summary>
    /// Exercises Shader Execution Reordering (SER)
    /// / the HitObject API - optixTraverse + optixReorder + optixInvoke as a 3-step
    /// alternative to a single optixTrace call, wired through the new
    /// OptixHitObject.Traverse/OptixReorder/OptixHitObject.Invoke device intrinsics
    /// (DeviceApi/OptixHitObjectTraverse.tt/OptixHitObjectInvoke.tt/OptixReorder.cs) -
    /// previously completely absent, no host or device surface at all.
    ///
    /// Every pixel renders the SAME hit TWICE - once via classic OptixTrace.Trace(...),
    /// once via OptixHitObject.Traverse(...) -> OptixReorder.Invoke() ->
    /// OptixHitObject.Invoke(...) - through the identical hitgroup/miss program, and
    /// compares the two results. SER only affects scheduling/performance, not
    /// correctness, so the two must be pixel-identical; any mismatch is drawn in
    /// bright magenta so a bug would be immediately obvious rather than silently
    /// averaged away. The "many divergent BSDFs" scene the plan called for is
    /// approximated by hashing each hit triangle's primitive index into a distinct
    /// color in the closest-hit program (OptixGetPrimitiveIndex.Value) - this project's
    /// shading is flat-color only, so per-triangle color divergence is what actually
    /// exercises "many different closest-hit outcomes reordered together", not a real
    /// BSDF evaluation.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public OptixDeviceView<uint> MismatchBuffer;
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

        private static uint Hash(uint x)
        {
            x = (x ^ 61u) ^ (x >> 16);
            x *= 9u;
            x ^= x >> 4;
            x *= 0x27d4eb2du;
            x ^= x >> 15;
            return x;
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
                // Unmissable error color - any correctness regression shows up as a
                // bright magenta pixel instead of being silently close-enough.
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

            long pixel = ix + (long)iy * launchParams.Width;
            launchParams.ColorBuffer[pixel] = rgba;
            launchParams.MismatchBuffer[pixel] = mismatch ? 1u : 0u;
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.08f);
            OptixPayloadInterop.SetFloat(1, 0.08f);
            OptixPayloadInterop.SetFloat(2, 0.12f);
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            // Hash the hit triangle's primitive index into a distinct color per
            // triangle - a stand-in for "many divergent BSDFs" (this project's
            // shading is flat-color only, see class doc comment).
            uint h = Hash(OptixGetPrimitiveIndex.Value * 2654435761u);
            float r = ((h >> 0) & 0xFFu) / 255f;
            float g = ((h >> 8) & 0xFFu) / 255f;
            float b = ((h >> 16) & 0xFFu) / 255f;
            OptixPayloadInterop.SetFloat(0, r);
            OptixPayloadInterop.SetFloat(1, g);
            OptixPayloadInterop.SetFloat(2, b);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private const int Width = 1024;
        private const int Height = 768;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private MemoryBuffer1D<uint, Stride1D.Dense> mismatchBuffer;
        private uint[] pixelHost;
        private uint[] mismatchHost;

        private BuiltAccelStructure accel;
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

        private int frameCount;
        private long totalMismatches;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample23: Shader Execution Reordering (Traverse/Reorder/Invoke)");
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
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(new byte[] { 0 });
            Console.WriteLine("SER pipeline (Traverse/Reorder/Invoke device intrinsics): PASS (compiled and linked successfully).");

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

            colorBuffer = accelerator.Allocate1D<uint>(Width * Height);
            mismatchBuffer = accelerator.Allocate1D<uint>(Width * Height);
            pixelHost = new uint[Width * Height];
            mismatchHost = new uint[Width * Height];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.MismatchBuffer = OptixDeviceView<uint>.From(mismatchBuffer);
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            // Startup correctness check: render once and count mismatches between
            // the classic-trace and SER-invoke paths - should be exactly zero.
            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);
            mismatchBuffer.CopyToCPU(mismatchHost);
            long startupMismatches = 0;
            foreach (var m in mismatchHost)
                startupMismatches += m;
            Console.WriteLine($"Startup frame: {startupMismatches} of {Width * Height} pixels mismatched between classic-trace and SER-invoke paths.");
            Console.WriteLine(startupMismatches == 0
                ? "Classic optixTrace vs. traverse/reorder/invoke: PASS (pixel-identical)"
                : "Classic optixTrace vs. traverse/reorder/invoke: FAIL (see magenta pixels)");

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Esc to quit.");
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

            launchParams.Width = Width;
            launchParams.Height = Height;
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

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);

            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, Width, Height,
                PixelFormat.Bgra, PixelType.UnsignedByte, pixelHost);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(glTexture);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
            {
                mismatchBuffer.CopyToCPU(mismatchHost);
                long mismatches = 0;
                foreach (var m in mismatchHost)
                    mismatches += m;
                totalMismatches += mismatches;
                Console.WriteLine($"[Sample23] frame {frameCount}, this-frame mismatches: {mismatches}");
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
            if (glTexture != 0)
                GL.DeleteTexture(glTexture);

            accel?.Dispose();
            colorBuffer?.Dispose();
            mismatchBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            Console.WriteLine($"[Sample23] total mismatches observed across the whole session: {totalMismatches}");

            base.OnUnload();
        }
    }
}
