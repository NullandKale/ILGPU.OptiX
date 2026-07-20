// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTexFootprint2D.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixTexFootprint2D/2DGrad/2DLod built-in
    /// functions - computes the texture-tile footprint a sample would touch, for
    /// demand-loaded (sparse) texturing. The 4-word result and the single-mip-level
    /// flag are returned by OptiX through pointers; this binding stages them in a
    /// small local-memory scratch block.
    /// </summary>
    /// <remarks>
    /// Unlike most of the DeviceApi this family is compile-verified only - no sample
    /// exercises demand-loaded texturing yet. The float arguments are passed as .b32
    /// words (bit casts), matching the SDK's own __float_as_uint plumbing.
    /// </remarks>
    [CLSCompliant(false)]
    public static class OptixTexFootprint2D
    {
        private static (uint X, uint Y, uint Z, uint W) ReadResult(
            ArrayView<uint> scratch,
            out uint singleMipLevel)
        {
            singleMipLevel = scratch[4];
            return (scratch[0], scratch[1], scratch[2], scratch[3]);
        }

        /// <summary>
        /// Provides the functionality of optixTexFootprint2D. <paramref name="texture"/>
        /// is the CUDA texture object handle; <paramref name="texInfo"/> packs the
        /// footprint granularity/options per the OptiX demand-texturing docs.
        /// </summary>
        public static (uint X, uint Y, uint Z, uint W) Footprint(
            ulong texture,
            uint texInfo,
            float x,
            float y,
            out uint singleMipLevel)
        {
            var scratch = LocalMemory.Allocate1D<uint>(5);
            ulong resultPtr = (ulong)scratch.BaseView.LoadEffectiveAddressAsPtr().ToInt64();
            ulong mipPtr = resultPtr + 4 * sizeof(uint);

            Input<ulong> _texture = texture;
            Input<uint> _texInfo = texInfo;
            Input<uint> _x = ILGPU.Interop.FloatAsInt(x);
            Input<uint> _y = ILGPU.Interop.FloatAsInt(y);
            Input<ulong> _mipPtr = mipPtr;
            Input<ulong> _resultPtr = resultPtr;
            CudaAsm.EmitRef(
                "call _optix_tex_footprint_2d_v2, (%0, %1, %2, %3, %4, %5);",
                ref _texture, ref _texInfo, ref _x, ref _y,
                ref _mipPtr, ref _resultPtr);

            return ReadResult(scratch, out singleMipLevel);
        }

        /// <summary>
        /// Provides the functionality of optixTexFootprint2DGrad.
        /// </summary>
        public static (uint X, uint Y, uint Z, uint W) FootprintGrad(
            ulong texture,
            uint texInfo,
            float x,
            float y,
            float dPdxX,
            float dPdxY,
            float dPdyX,
            float dPdyY,
            bool coarse,
            out uint singleMipLevel)
        {
            var scratch = LocalMemory.Allocate1D<uint>(5);
            ulong resultPtr = (ulong)scratch.BaseView.LoadEffectiveAddressAsPtr().ToInt64();
            ulong mipPtr = resultPtr + 4 * sizeof(uint);

            Input<ulong> _texture = texture;
            Input<uint> _texInfo = texInfo;
            Input<uint> _x = ILGPU.Interop.FloatAsInt(x);
            Input<uint> _y = ILGPU.Interop.FloatAsInt(y);
            Input<uint> _dPdxX = ILGPU.Interop.FloatAsInt(dPdxX);
            Input<uint> _dPdxY = ILGPU.Interop.FloatAsInt(dPdxY);
            Input<uint> _dPdyX = ILGPU.Interop.FloatAsInt(dPdyX);
            Input<uint> _dPdyY = ILGPU.Interop.FloatAsInt(dPdyY);
            Input<uint> _coarse = coarse ? 1u : 0u;
            Input<ulong> _mipPtr = mipPtr;
            Input<ulong> _resultPtr = resultPtr;
            CudaAsm.EmitRef(
                "call _optix_tex_footprint_2d_grad_v2, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10);",
                ref _texture, ref _texInfo, ref _x, ref _y,
                ref _dPdxX, ref _dPdxY, ref _dPdyX, ref _dPdyY,
                ref _coarse, ref _mipPtr, ref _resultPtr);

            return ReadResult(scratch, out singleMipLevel);
        }

        /// <summary>
        /// Provides the functionality of optixTexFootprint2DLod.
        /// </summary>
        public static (uint X, uint Y, uint Z, uint W) FootprintLod(
            ulong texture,
            uint texInfo,
            float x,
            float y,
            float level,
            bool coarse,
            out uint singleMipLevel)
        {
            var scratch = LocalMemory.Allocate1D<uint>(5);
            ulong resultPtr = (ulong)scratch.BaseView.LoadEffectiveAddressAsPtr().ToInt64();
            ulong mipPtr = resultPtr + 4 * sizeof(uint);

            Input<ulong> _texture = texture;
            Input<uint> _texInfo = texInfo;
            Input<uint> _x = ILGPU.Interop.FloatAsInt(x);
            Input<uint> _y = ILGPU.Interop.FloatAsInt(y);
            Input<uint> _level = ILGPU.Interop.FloatAsInt(level);
            Input<uint> _coarse = coarse ? 1u : 0u;
            Input<ulong> _mipPtr = mipPtr;
            Input<ulong> _resultPtr = resultPtr;
            CudaAsm.EmitRef(
                "call _optix_tex_footprint_2d_lod_v2, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7);",
                ref _texture, ref _texInfo, ref _x, ref _y,
                ref _level, ref _coarse, ref _mipPtr, ref _resultPtr);

            return ReadResult(scratch, out singleMipLevel);
        }
    }
}
