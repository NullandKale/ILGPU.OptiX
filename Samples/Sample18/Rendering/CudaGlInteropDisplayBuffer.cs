using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Graphics.OpenGL4;
using System;
using System.Diagnostics;

namespace Sample18
{
    /// <summary>
    /// Zero-copy CUDA-GL interop for a full-window BGRA display buffer - same
    /// register/map/GetMappedPointer/unmap pattern as Sample13's
    /// Rendering/CudaGlInteropDisplayBuffer.cs, copied per-sample rather than
    /// shared from Src/ILGPU.OptiX (that library deliberately stays
    /// windowing-library-agnostic - no OpenTK/GL dependency - so anything that
    /// calls GL.* lives in the sample, not the core library).
    ///
    /// Unlike Sample13 (one raygen pipeline covering the whole window), this
    /// sample's window is N simultaneous tiles, each its own OptiX pipeline. All N
    /// pipelines write into different offset/stride windows of this SAME buffer
    /// every frame (no per-tile buffer, no host round-trip) - see RenderWindow's
    /// per-tile LaunchParams.XOffset/Stride fields.
    /// </summary>
    public sealed class CudaGlInteropDisplayBuffer : MemoryBuffer
    {
        readonly int width;
        readonly int height;
        readonly int byteCount;

        IntPtr cudaResource;
        State state;

        public int GlBufferHandle { get; private set; }
        public int GlTextureHandle { get; private set; }

        public CudaGlInteropDisplayBuffer(int width, int height, CudaAccelerator accelerator)
            : base(accelerator, width * height * 4, 1)
        {
            this.width = width;
            this.height = height;
            byteCount = width * height * 4;

            GlBufferHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, GlBufferHandle);
            GL.BufferData(BufferTarget.PixelUnpackBuffer, byteCount, IntPtr.Zero, BufferUsageHint.StreamDraw);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

            GlTextureHandle = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, GlTextureHandle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, width, height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            CudaException.ThrowIfFailed(CudaGlInterop.RegisterBuffer(
                out cudaResource,
                GlBufferHandle,
                (uint)CudaGraphicsMapFlags.WriteDiscard)); // CUDA only ever writes this buffer

            state = State.AvailableForGl;
        }

        public void MapCuda(CudaStream stream)
        {
            if (state != State.AvailableForGl)
                return;

            unsafe
            {
                fixed (IntPtr* pResource = &cudaResource)
                {
                    CudaException.ThrowIfFailed(CudaGlInterop.MapResources(1, new IntPtr(pResource), stream.StreamPtr));
                }
            }
            state = State.MappedToCuda;
        }

        public ArrayView<byte> GetCudaArrayView()
        {
            if (state != State.MappedToCuda)
                throw new InvalidOperationException("Buffer must be mapped to CUDA before accessing.");

            CudaException.ThrowIfFailed(CudaGlInterop.GetMappedPointer(out var devicePtr, out var mappedSize, cudaResource));
            Trace.Assert(mappedSize == byteCount);
            NativePtr = devicePtr;

            return AsArrayView<byte>(0, byteCount);
        }

        public void UnmapCuda(CudaStream stream)
        {
            if (state != State.MappedToCuda)
                return;

            unsafe
            {
                fixed (IntPtr* pResource = &cudaResource)
                {
                    CudaException.ThrowIfFailed(CudaGlInterop.UnmapResources(1, new IntPtr(pResource), stream.StreamPtr));
                }
            }
            NativePtr = IntPtr.Zero;
            state = State.AvailableForGl;
        }

        // GPU-internal PBO -> texture blit (no CPU copy) - call once per frame after
        // UnmapCuda, before drawing the fullscreen quad.
        public void BlitToTexture()
        {
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, GlBufferHandle);
            GL.BindTexture(TextureTarget.Texture2D, GlTextureHandle);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, width, height,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.BindTexture(TextureTarget.Texture2D, 0);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
        }

        public static void CudaMemSet<T>(CudaStream stream, byte value, in ArrayView<T> targetView)
            where T : unmanaged
        {
            if (stream is null)
                throw new ArgumentNullException(nameof(stream));
            if (targetView.GetAcceleratorType() != AcceleratorType.Cuda)
                throw new NotSupportedException();

            var binding = stream.Accelerator.BindScoped();
            CudaException.ThrowIfFailed(CudaAPI.CurrentAPI.Memset(
                targetView.LoadEffectiveAddressAsPtr(), value, new IntPtr(targetView.LengthInBytes), stream));
            binding.Recover();
        }

        public static void CudaCopy<T>(CudaStream stream, in ArrayView<T> sourceView, in ArrayView<T> targetView)
            where T : unmanaged
        {
            if (stream is null)
                throw new ArgumentNullException(nameof(stream));

            using var binding = stream.Accelerator.BindScoped();

            if (sourceView.GetAcceleratorType() == AcceleratorType.OpenCL || targetView.GetAcceleratorType() == AcceleratorType.OpenCL)
                throw new NotSupportedException();

            CudaException.ThrowIfFailed(CudaAPI.CurrentAPI.MemcpyAsync(
                targetView.LoadEffectiveAddressAsPtr(),
                sourceView.LoadEffectiveAddressAsPtr(),
                new IntPtr(targetView.LengthInBytes),
                stream));
        }

        protected override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> targetView) =>
            CudaMemSet(stream as CudaStream, value, targetView);

        protected override void CopyFrom(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView) =>
            CudaCopy(stream as CudaStream, sourceView, targetView);

        protected override void CopyTo(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView) =>
            CudaCopy(stream as CudaStream, sourceView, targetView);

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                if (cudaResource != IntPtr.Zero)
                {
                    CudaGlInterop.UnregisterResource(cudaResource);
                    cudaResource = IntPtr.Zero;
                }
                if (GlTextureHandle != 0)
                {
                    GL.DeleteTexture(GlTextureHandle);
                    GlTextureHandle = 0;
                }
                if (GlBufferHandle != 0)
                {
                    GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
                    GL.DeleteBuffer(GlBufferHandle);
                    GlBufferHandle = 0;
                }
            }
            base.Dispose();
        }

        enum State
        {
            AvailableForGl,
            MappedToCuda
        }
    }
}
