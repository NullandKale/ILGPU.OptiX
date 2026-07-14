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

namespace Sample19
{
    /// <summary>
    /// Exercises rigid-body motion blur via a
    /// matrix motion transform (optixConvertPointerToTraversableHandle +
    /// OptixMatrixMotionTransform, built through the new
    /// OptixMotionTransformBuilder.BuildMatrixMotionTransform) wrapping a static GAS.
    ///
    /// The mesh spins ~90 degrees and translates while OptixTrace's rayTime argument
    /// (previously always hardcoded to 0 by every existing sample) sweeps back and
    /// forth between the transform's two motion keys, driven by real time - a live,
    /// continuously animating rigid-body motion demo, not a single static blurred
    /// frame. Space toggles the animation on/off (frozen at the first motion key) to
    /// contrast moving vs. static.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public int Width;
            public int Height;
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

            long pixel = ix + (long)iy * launchParams.Width;
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

        private const int Width = 1024;
        private const int Height = 768;
        private const float TranslateDistance = 2.5f;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private uint[] pixelHost;

        private BuiltAccelStructure gas;
        private BuiltMotionTransform motionTransform;
        private BuiltAccelStructure ias;
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

        private bool motionEnabled = true;
        private bool spaceWasDown;
        private readonly System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();

        private int frameCount;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample19: rigid-body motion blur (matrix motion transform)");
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

            using var vertexBuffer = accelerator.Allocate1D(vertices);
            using var indexBuffer = accelerator.Allocate1D(indices);

            Console.WriteLine("Building static GAS...");
            gas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(vertexBuffer, indexBuffer)
                .AllowCompaction()
                .Build();

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
            // Center the camera between the two motion-key positions, and widen the
            // orbit distance so the whole translate range stays framed.
            center += new Vector3(TranslateDistance * 0.5f, 0f, 0f);
            SetupCamera(center, radius + TranslateDistance * 0.5f);

            colorBuffer = accelerator.Allocate1D<uint>(Width * Height);
            pixelHost = new uint[Width * Height];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.Traversable = unchecked((ulong)ias.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            Console.WriteLine("Motion transform + IAS built successfully: PASS.");
            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Space toggles motion, Esc to quit.");
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
            sceneCenter = center;
            orbitDistance = radius * 1.8f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.6f, 0.5f, 1.4f));
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

            bool spaceDown = KeyboardState.IsKeyDown(Keys.Space);
            if (spaceDown && !spaceWasDown)
            {
                motionEnabled = !motionEnabled;
                Console.WriteLine($"[Motion] {(motionEnabled ? "enabled" : "disabled (frozen at key 0)")}");
            }
            spaceWasDown = spaceDown;

            UpdateMouseOrbit();
        }

        protected override void OnRenderFrame(FrameEventArgs args)
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
                Console.WriteLine($"[Sample19] frame {frameCount}, rayTime={launchParams.RayTime:0.00}");
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

            ias?.Dispose();
            motionTransform?.Dispose();
            gas?.Dispose();
            colorBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
