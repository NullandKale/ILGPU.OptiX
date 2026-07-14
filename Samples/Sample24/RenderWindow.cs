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
using ILGPU.OptiX.CooperativeVectors;
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
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Half = ILGPU.Half;

namespace Sample24
{
    /// <summary>
    /// Exercises Cooperative Vectors - a tiny 2-layer
    /// MLP ("neural material") evaluated per-pixel entirely inside the raygen program via
    /// <see cref="OptixCoopVec.MatVecMul"/>/<see cref="OptixCoopVec.Tanh"/>, using weight
    /// matrices uploaded row-major on the host and converted to the driver's
    /// InferencingOptimal layout via <see cref="OptixCoopVecMatrixBuilder.ConvertMatrix"/>.
    /// The network's 3 inputs are the hit's barycentric coordinates plus a per-triangle
    /// pseudo-random seed (hashed primitive index) - not photorealistic, just enough
    /// spatially-varying signal to prove the host conversion + device matvecmul/activation
    /// path actually runs and produces a real, non-constant result per pixel (same
    /// "correctness over performance" scope as every other item-2-9 sample).
    ///
    /// Weight/bias/scratch buffers are ordinary ILGPU device buffers; their addresses are
    /// obtained once on the host via <see cref="OptixCoopVecBufferExtensions.GetDeviceAddress"/>
    /// and carried through LaunchParams as plain ulong fields (the same way an
    /// OptixTraversableHandle already flows through one) - see OptixCoopVec.cs's class doc
    /// comment for why device code never derives these addresses itself.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private const uint InputSize = 3;
        private const uint HiddenSize = 8;
        private const uint OutputSize = 3;

        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public OptixDeviceView<Half> InputScratch;
            public OptixDeviceView<Half> OutputScratch;
            public ulong InputScratchBase;
            public ulong HiddenScratchBase;
            public ulong OutputScratchBase;
            public ulong Bias1Base;
            public ulong Bias2Base;
            public ulong WeightsBase;
            public uint Weight2OffsetInBytes;
            public ulong HiddenGainBase;
            public ulong OutputScaleBase;
            public ulong OutputBiasBase;
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
            public uint P0, P1, P2, P3;
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
                1,
                0,
                ref p0,
                ref p1,
                ref p2,
                ref p3);

            long pixel = ix + (long)iy * launchParams.Width;
            bool hit = Interop.IntAsFloat(p0) > 0.5f;

            uint ru, gu, bu;
            if (hit)
            {
                float bary0 = Interop.IntAsFloat(p1);
                float bary1 = Interop.IntAsFloat(p2);
                float seed = Interop.IntAsFloat(p3);

                long inputBase = pixel * InputSize;
                launchParams.InputScratch[inputBase + 0] = (Half)(bary0 * 2f - 1f);
                launchParams.InputScratch[inputBase + 1] = (Half)(bary1 * 2f - 1f);
                launchParams.InputScratch[inputBase + 2] = (Half)seed;

                uint halfSize = (uint)Interop.SizeOf<Half>();
                ulong inputAddr = launchParams.InputScratchBase + (ulong)pixel * InputSize * halfSize;
                ulong hiddenAddr = launchParams.HiddenScratchBase + (ulong)pixel * HiddenSize * halfSize;
                ulong outputAddr = launchParams.OutputScratchBase + (ulong)pixel * OutputSize * halfSize;

                // Layer 1: hidden = tanh(W1 * input + bias1). Both layers' weights live
                // in one converted buffer (packed by matrixOffsetInBytes), matching the
                // real OptiX SDK reference sample's own layout
                // (SDK/optixNeuralTexture/optixNeuralTexture.cpp's convertWeights) -
                // one optixCoopVecMatrixConvert call for the whole network, not one per
                // layer. Uses the concrete N8_K3/S8/S3 overloads (OptixCoopVec.cs is
                // T4-generated with the shape/type baked as PTX-literal constants - see
                // its class doc comment for why a runtime N/K/elementType parameter
                // doesn't compile on real hardware).
                OptixCoopVec.MatVecMul_N8_K3(
                    inputAddr, launchParams.WeightsBase, 0, launchParams.Bias1Base, 0, hiddenAddr);
                OptixCoopVec.Tanh_S8(hiddenAddr, hiddenAddr);
                // Per-neuron gain (elementwise Mul) - a real second use of the "_ptr" op2
                // path beyond the activation itself, applied in place.
                OptixCoopVec.Mul_S8(hiddenAddr, launchParams.HiddenGainBase, hiddenAddr);

                // Layer 2: output = tanh(W2 * hidden + bias2)
                OptixCoopVec.MatVecMul_N3_K8(
                    hiddenAddr, launchParams.WeightsBase, launchParams.Weight2OffsetInBytes, launchParams.Bias2Base, 0, outputAddr);
                OptixCoopVec.Tanh_S3(outputAddr, outputAddr);
                // Range remap [-1,1] -> [0,1] as a single fused multiply-add device call
                // (output = output * 0.5 + 0.5) instead of per-channel host-side math.
                OptixCoopVec.FFma_S3(
                    outputAddr, launchParams.OutputScaleBase, launchParams.OutputBiasBase, outputAddr);

                long outputBase = pixel * OutputSize;
                float r = launchParams.OutputScratch[outputBase + 0];
                float g = launchParams.OutputScratch[outputBase + 1];
                float b = launchParams.OutputScratch[outputBase + 2];
                ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
                gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
                bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);
            }
            else
            {
                ru = 20; gu = 20; bu = 30;
            }

            uint rgba = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);
            launchParams.ColorBuffer[pixel] = rgba;
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0f);
            OptixPayloadInterop.SetFloat(1, 0f);
            OptixPayloadInterop.SetFloat(2, 0f);
            OptixPayloadInterop.SetFloat(3, 0f);
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            var (bu, bv) = OptixGetTriangleBarycentrics.Value;
            uint h = Hash(OptixGetPrimitiveIndex.Value * 2654435761u);
            float seed = ((h & 0xFFFFu) / 65535f) * 2f - 1f;

            OptixPayloadInterop.SetFloat(0, 1f);
            OptixPayloadInterop.SetFloat(1, bu);
            OptixPayloadInterop.SetFloat(2, bv);
            OptixPayloadInterop.SetFloat(3, seed);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private const int Width = 1024;
        private const int Height = 768;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> inputScratchBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> hiddenScratchBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> outputScratchBuffer;
        private MemoryBuffer1D<byte, Stride1D.Dense> weightsRawBuffer;
        private MemoryBuffer1D<byte, Stride1D.Dense> weightsBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> bias1Buffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> bias2Buffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> hiddenGainBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> outputScaleBuffer;
        private MemoryBuffer1D<Half, Stride1D.Dense> outputBiasBuffer;
        private uint[] pixelHost;

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

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        // Deterministic small pseudo-random weights - this is a correctness probe for
        // the coop-vec API surface, not a trained network (no training data/loss exists).
        // Half, not float - Float16 is the only element type optixCoopVecMatMul fully
        // supports input-to-output without mixing in an integer type (see the OptiX 9.0
        // Programming Guide's "Neural Rendering with Cooperative Vectors" chapter's
        // supported-type-combination table; confirmed the hard way via a real GPU
        // compile error, "output vector type is of unsupported type (0x2A03)").
        private static Half[] RandomWeights(Random rng, int count, float scale)
        {
            var result = new Half[count];
            for (int i = 0; i < count; i++)
                result[i] = (Half)((float)(rng.NextDouble() * 2.0 - 1.0) * scale);
            return result;
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample24: Cooperative Vectors (neural material)");
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
                .MaxTraceDepth(1));
            pipeline.SetHitRecords<byte>(new byte[] { 0 });
            Console.WriteLine("Pipeline (raygen-side OptixCoopVec.MatVecMul/Tanh): PASS (compiled and linked successfully).");

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
            inputScratchBuffer = accelerator.Allocate1D<Half>(Width * Height * (int)InputSize);
            hiddenScratchBuffer = accelerator.Allocate1D<Half>(Width * Height * (int)HiddenSize);
            outputScratchBuffer = accelerator.Allocate1D<Half>(Width * Height * (int)OutputSize);
            pixelHost = new uint[Width * Height];

            var deviceContext = rayTracer.DeviceContext;

            Console.WriteLine("Building + converting cooperative-vector weight matrices to InferencingOptimal layout...");
            var rng = new Random(1234);

            var layers = new[]
            {
                (N: HiddenSize, K: InputSize, RowMajorWeights: RandomWeights(rng, (int)(HiddenSize * InputSize), 0.7f)),
                (N: OutputSize, K: HiddenSize, RowMajorWeights: RandomWeights(rng, (int)(OutputSize * HiddenSize), 0.7f)),
            };
            uint[] weightOffsetsInBytes;
            (weightsRawBuffer, weightsBuffer, weightOffsetsInBytes) = BuildAndConvertNetwork(deviceContext, accelerator, layers);

            bias1Buffer = accelerator.Allocate1D(RandomWeights(rng, (int)HiddenSize, 0.2f));
            bias2Buffer = accelerator.Allocate1D(RandomWeights(rng, (int)OutputSize, 0.2f));

            // Per-neuron gain applied via OptixCoopVec.Mul, and the [-1,1]->[0,1] range
            // remap applied via OptixCoopVec.FFma - both plain constant vectors, not
            // trained, same "correctness probe" scope as the weights themselves.
            var hiddenGain = new Half[HiddenSize];
            for (int i = 0; i < hiddenGain.Length; i++)
                hiddenGain[i] = (Half)(0.7f + 0.3f * ((i % 3) / 2f));
            hiddenGainBuffer = accelerator.Allocate1D(hiddenGain);

            var halfConstant = new Half[OutputSize];
            Array.Fill(halfConstant, (Half)0.5f);
            outputScaleBuffer = accelerator.Allocate1D(halfConstant);
            outputBiasBuffer = accelerator.Allocate1D(halfConstant);

            accelerator.Synchronize();
            Console.WriteLine($"Network (2 layers, {HiddenSize}x{InputSize} + {OutputSize}x{HiddenSize}) converted in one " +
                $"optixCoopVecMatrixConvert call: {weightsBuffer.LengthInBytes} bytes InferencingOptimal " +
                $"(layer offsets {weightOffsetsInBytes[0]}, {weightOffsetsInBytes[1]}). Matrix convert: PASS.");

            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.InputScratch = OptixDeviceView<Half>.From(inputScratchBuffer);
            launchParams.OutputScratch = OptixDeviceView<Half>.From(outputScratchBuffer);
            launchParams.InputScratchBase = inputScratchBuffer.GetDeviceAddress();
            launchParams.HiddenScratchBase = hiddenScratchBuffer.GetDeviceAddress();
            launchParams.OutputScratchBase = outputScratchBuffer.GetDeviceAddress();
            launchParams.WeightsBase = weightsBuffer.GetDeviceAddress();
            launchParams.Weight2OffsetInBytes = weightOffsetsInBytes[1];
            launchParams.Bias1Base = bias1Buffer.GetDeviceAddress();
            launchParams.Bias2Base = bias2Buffer.GetDeviceAddress();
            launchParams.HiddenGainBase = hiddenGainBuffer.GetDeviceAddress();
            launchParams.OutputScaleBase = outputScaleBuffer.GetDeviceAddress();
            launchParams.OutputBiasBase = outputBiasBuffer.GetDeviceAddress();
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);
            const uint backgroundRgba = 0xff000000u | (30u << 0) | (20u << 8) | (20u << 16);
            int nonBackground = 0;
            foreach (var p in pixelHost)
                if (p != backgroundRgba)
                    nonBackground++;
            Console.WriteLine($"Startup frame: {nonBackground} of {Width * Height} pixels shaded by the neural material (mesh hit).");

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Esc to quit.");
        }

        /// <summary>
        /// Packs every layer of a multi-layer network into one row-major buffer and
        /// converts the whole network to InferencingOptimal in a single
        /// <see cref="OptixCoopVecMatrixBuilder.ConvertMatrices"/> call - the same
        /// per-layer-offset packing + one-call-per-network pattern the real OptiX SDK
        /// reference sample uses (SDK/optixNeuralTexture/optixNeuralTexture.cpp's
        /// convertWeights: source offsets from the running sum of source layer sizes,
        /// destination offsets from the running sum of destination layer sizes -
        /// independently, since row-major and InferencingOptimal don't need to agree on
        /// per-layer size).
        /// </summary>
        private static (MemoryBuffer1D<byte, Stride1D.Dense> Raw, MemoryBuffer1D<byte, Stride1D.Dense> Converted, uint[] DstOffsetsInBytes) BuildAndConvertNetwork(
            OptixDeviceContext deviceContext, CudaAccelerator accelerator, (uint N, uint K, Half[] RowMajorWeights)[] layers)
        {
            var srcDescriptions = new OptixCoopVecMatrixDescription[layers.Length];
            var dstDescriptions = new OptixCoopVecMatrixDescription[layers.Length];
            uint srcOffset = 0;
            uint dstOffset = 0;
            for (int i = 0; i < layers.Length; i++)
            {
                uint srcSize = deviceContext.ComputeMatrixSizeInBytes(layers[i].N, layers[i].K, OptixCoopVecElemType.Float16, OptixCoopVecMatrixLayout.RowMajor);
                uint dstSize = deviceContext.ComputeMatrixSizeInBytes(layers[i].N, layers[i].K, OptixCoopVecElemType.Float16, OptixCoopVecMatrixLayout.InferencingOptimal);

                srcDescriptions[i] = new OptixCoopVecMatrixDescription
                {
                    N = layers[i].N,
                    K = layers[i].K,
                    OffsetInBytes = srcOffset,
                    ElementType = OptixCoopVecElemType.Float16,
                    Layout = OptixCoopVecMatrixLayout.RowMajor,
                    RowColumnStrideInBytes = 0,
                    SizeInBytes = srcSize,
                };
                dstDescriptions[i] = new OptixCoopVecMatrixDescription
                {
                    N = layers[i].N,
                    K = layers[i].K,
                    OffsetInBytes = dstOffset,
                    ElementType = OptixCoopVecElemType.Float16,
                    Layout = OptixCoopVecMatrixLayout.InferencingOptimal,
                    RowColumnStrideInBytes = 0,
                    SizeInBytes = dstSize,
                };

                srcOffset += srcSize;
                dstOffset += dstSize;
            }

            var rawBytes = new byte[srcOffset];
            for (int i = 0; i < layers.Length; i++)
            {
                var weightBytes = MemoryMarshal.AsBytes<Half>(layers[i].RowMajorWeights);
                weightBytes.CopyTo(rawBytes.AsSpan((int)srcDescriptions[i].OffsetInBytes));
            }

            var rawBuffer = accelerator.Allocate1D(rawBytes);
            var convertedBuffer = accelerator.Allocate1D<byte>(dstOffset);

            deviceContext.ConvertMatrices(
                accelerator.DefaultStream,
                numNetworks: 1,
                srcDescriptions, rawBuffer, inputNetworkStrideInBytes: 0,
                dstDescriptions, convertedBuffer, outputNetworkStrideInBytes: 0);

            return (rawBuffer, convertedBuffer, dstDescriptions.Select(d => d.OffsetInBytes).ToArray());
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
                Console.WriteLine($"[Sample24] frame {frameCount}");
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
            inputScratchBuffer?.Dispose();
            hiddenScratchBuffer?.Dispose();
            outputScratchBuffer?.Dispose();
            weightsRawBuffer?.Dispose();
            weightsBuffer?.Dispose();
            bias1Buffer?.Dispose();
            bias2Buffer?.Dispose();
            hiddenGainBuffer?.Dispose();
            outputScaleBuffer?.Dispose();
            outputBiasBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
