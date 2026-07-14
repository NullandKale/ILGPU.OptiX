// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: CudaTextureObject.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Cuda
{
    // Wraps a bindless CUDA texture object (cudaTextureObject_t) backed by a CUDA array,
    // for OptiX materials that sample a 2D RGBA8 texture. Device-side sampling goes
    // through CudaTex2D.Sample, passing the handle as a plain ulong (see CudaTex2D.cs).
    // Neither ILGPU nor ILGPU.OptiX expose the CUDA driver's texture-object entry
    // points (confirmed by inspecting the ILGPU 1.5.3 assembly - no CudaArray/
    // CudaTextureObject/cuTexObjectCreate binding exists), so this P/Invokes
    // nvcuda.dll directly, matching the pattern OptixAPI.cs uses for OptiX's own
    // driver calls. Entry point names (cuArrayCreate_v2, cuMemcpy2D_v2) were verified
    // against the exports of the installed nvcuda.dll, since the versioned suffix is
    // normally applied by a header macro that isn't available to P/Invoke.
    [CLSCompliant(false)]
    public sealed class CudaTextureObject : DisposeBase
    {
        private const int CU_AD_FORMAT_UNSIGNED_INT8 = 0x01;
        private const int CU_MEMORYTYPE_HOST = 0x01;
        private const int CU_MEMORYTYPE_ARRAY = 0x03;
        private const int CU_RESOURCE_TYPE_ARRAY = 0x00;
        private const int CU_TR_ADDRESS_MODE_WRAP = 0;
        private const int CU_TR_FILTER_MODE_LINEAR = 1;
        private const uint CU_TRSF_NORMALIZED_COORDINATES = 0x02;

        [StructLayout(LayoutKind.Sequential)]
        private struct CUDA_ARRAY_DESCRIPTOR
        {
            public ulong Width;
            public ulong Height;
            public int Format;
            public uint NumChannels;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CUDA_MEMCPY2D
        {
            public ulong srcXInBytes;
            public ulong srcY;
            public int srcMemoryType;
            public IntPtr srcHost;
            public ulong srcDevice;
            public IntPtr srcArray;
            public ulong srcPitch;

            public ulong dstXInBytes;
            public ulong dstY;
            public int dstMemoryType;
            public IntPtr dstHost;
            public ulong dstDevice;
            public IntPtr dstArray;
            public ulong dstPitch;

            public ulong WidthInBytes;
            public ulong Height;
        }

        // The real CUDA_RESOURCE_DESC has a 128-byte union of resource-type-specific
        // sub-structs (the "reserved" branch, int[32], is the largest); only the
        // "array" branch is used here, but the driver still expects the full-size
        // struct - explicit layout reserves the whole union rather than risk an
        // out-of-bounds read past a too-small managed struct.
        [StructLayout(LayoutKind.Explicit, Size = 144)]
        private struct CUDA_RESOURCE_DESC
        {
            [FieldOffset(0)] public int resType;
            [FieldOffset(8)] public IntPtr hArray;
            [FieldOffset(136)] public uint flags;
        }

        // Full native size (104 bytes) reserved even though only the leading fields
        // are set, for the same out-of-bounds-read reason as CUDA_RESOURCE_DESC.
        [StructLayout(LayoutKind.Sequential)]
        private struct CUDA_TEXTURE_DESC
        {
            public int AddressMode0;
            public int AddressMode1;
            public int AddressMode2;
            public int FilterMode;
            public uint Flags;
            public uint MaxAnisotropy;
            public int MipmapFilterMode;
            public float MipmapLevelBias;
            public float MinMipmapLevelClamp;
            public float MaxMipmapLevelClamp;
            public float BorderColor0, BorderColor1, BorderColor2, BorderColor3;
            private int reserved0, reserved1, reserved2, reserved3, reserved4, reserved5,
                reserved6, reserved7, reserved8, reserved9, reserved10, reserved11;
        }

        [DllImport("nvcuda.dll")]
        private static extern int cuCtxGetCurrent(out IntPtr pctx);

        [DllImport("nvcuda.dll")]
        private static extern int cuCtxSetCurrent(IntPtr ctx);

        [DllImport("nvcuda.dll")]
        private static extern int cuArrayCreate_v2(out IntPtr pHandle, ref CUDA_ARRAY_DESCRIPTOR pAllocateArray);

        [DllImport("nvcuda.dll")]
        private static extern int cuArrayDestroy(IntPtr hArray);

        [DllImport("nvcuda.dll")]
        private static extern int cuMemcpy2D_v2(ref CUDA_MEMCPY2D pCopy);

        [DllImport("nvcuda.dll")]
        private static extern int cuTexObjectCreate(
            out ulong pTexObject,
            ref CUDA_RESOURCE_DESC pResDesc,
            ref CUDA_TEXTURE_DESC pTexDesc,
            IntPtr pResViewDesc);

        [DllImport("nvcuda.dll")]
        private static extern int cuTexObjectDestroy(ulong texObject);

        private IntPtr array;
        private readonly int width;
        private readonly int height;

        // Captured at construction (cuCtxGetCurrent) and re-asserted (cuCtxSetCurrent)
        // at the start of every later driver call - CUDA driver contexts are
        // thread-affinitized (current-per-OS-thread, not per-process), so a raw
        // nvcuda.dll call from a different thread than the one that created this
        // object (e.g. Update() called from Sample13's dedicated render thread, while
        // construction happened on the UI thread during SwitchToScene) would otherwise
        // fail with CUDA_ERROR_INVALID_CONTEXT (201) - ILGPU's own accelerator calls
        // self-manage this, but this class's direct P/Invokes don't unless told to.
        private IntPtr context;

        /// <summary>
        /// The bindless CUDA texture object handle (cudaTextureObject_t), to be passed
        /// into device code as a plain ulong - see CudaTex2D.Sample.
        /// </summary>
        public ulong TextureObject { get; private set; }

        /// <summary>
        /// Creates a bindless CUDA texture object from RGBA8 pixel data. Assumes the
        /// calling thread's current CUDA context is the one the texture will be
        /// sampled from (true for the single-accelerator samples in this repo) - that
        /// context is captured here and re-asserted on every subsequent driver call
        /// regardless of which thread makes it (see the `context` field comment).
        /// </summary>
        /// <param name="rgba">Tightly-packed RGBA8 pixel data, row-major, top-to-bottom.</param>
        /// <param name="width">Texture width in pixels.</param>
        /// <param name="height">Texture height in pixels.</param>
        public unsafe CudaTextureObject(byte[] rgba, int width, int height)
        {
            if (rgba == null)
                throw new ArgumentNullException(nameof(rgba));
            if (rgba.Length != width * height * 4)
                throw new ArgumentException("RGBA data length does not match width * height * 4.", nameof(rgba));

            this.width = width;
            this.height = height;
            CudaCheck(cuCtxGetCurrent(out context));

            var arrayDesc = new CUDA_ARRAY_DESCRIPTOR
            {
                Width = (ulong)width,
                Height = (ulong)height,
                Format = CU_AD_FORMAT_UNSIGNED_INT8,
                NumChannels = 4
            };
            CudaCheck(cuArrayCreate_v2(out array, ref arrayDesc));

            UploadPixels(rgba);

            var resDesc = new CUDA_RESOURCE_DESC
            {
                resType = CU_RESOURCE_TYPE_ARRAY,
                hArray = array,
                flags = 0
            };
            var texDesc = new CUDA_TEXTURE_DESC
            {
                AddressMode0 = CU_TR_ADDRESS_MODE_WRAP,
                AddressMode1 = CU_TR_ADDRESS_MODE_WRAP,
                AddressMode2 = CU_TR_ADDRESS_MODE_WRAP,
                FilterMode = CU_TR_FILTER_MODE_LINEAR,
                Flags = CU_TRSF_NORMALIZED_COORDINATES
            };

            CudaCheck(cuTexObjectCreate(out var texObject, ref resDesc, ref texDesc, IntPtr.Zero));
            TextureObject = texObject;
        }

        private unsafe void UploadPixels(byte[] rgba)
        {
            CudaCheck(cuCtxSetCurrent(context));
            fixed (byte* src = rgba)
            {
                var copy = new CUDA_MEMCPY2D
                {
                    srcMemoryType = CU_MEMORYTYPE_HOST,
                    srcHost = new IntPtr(src),
                    srcPitch = (ulong)(width * 4),
                    dstMemoryType = CU_MEMORYTYPE_ARRAY,
                    dstArray = array,
                    WidthInBytes = (ulong)(width * 4),
                    Height = (ulong)height
                };
                CudaCheck(cuMemcpy2D_v2(ref copy));
            }
        }

        /// <summary>
        /// Re-uploads new RGBA8 pixel data into the existing CUDA array, reusing the
        /// same array and texture object (so <see cref="TextureObject"/> stays valid at
        /// its already-baked-into-the-SBT handle value) - for video/dynamic textures
        /// refreshed once per rendered frame. Dimensions must match the size this
        /// object was constructed with.
        /// </summary>
        /// <param name="rgba">Tightly-packed RGBA8 pixel data, row-major, top-to-bottom, width*height*4 bytes.</param>
        public void Update(byte[] rgba)
        {
            if (rgba == null)
                throw new ArgumentNullException(nameof(rgba));
            if (rgba.Length != width * height * 4)
                throw new ArgumentException("RGBA data length does not match width * height * 4.", nameof(rgba));

            UploadPixels(rgba);
        }

        private static void CudaCheck(int cuResult)
        {
            if (cuResult != 0)
                throw new InvalidOperationException($"CUDA driver call failed with CUresult {cuResult}.");
        }

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (TextureObject != 0)
                {
                    cuTexObjectDestroy(TextureObject);
                    TextureObject = 0;
                }
                if (array != IntPtr.Zero)
                {
                    cuArrayDestroy(array);
                    array = IntPtr.Zero;
                }
            }
            base.Dispose(disposing);
        }
    }
}
