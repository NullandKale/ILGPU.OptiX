// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDenoiserOptions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1028 // Enum Storage should be Int32
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.Denoising
{
    public enum OptixDenoiserModelKind
    {
        Aov = 0x2324,
        TemporalAov = 0x2326,
        Upscale2x = 0x2327,
        TemporalUpscale2x = 0x2328,

        // Deprecated by the SDK - internally mapped to OptixDenoiserModelKind.Aov -
        // but this is what the optix7course reference examples (built against an
        // older SDK) use, so it's kept here rather than only exposing the modern name.
        Ldr = 0x2322,
        Hdr = 0x2323,
        Temporal = 0x2325,
    }

    public enum OptixDenoiserAlphaMode
    {
        Copy = 0,
        Denoise = 1,
    }

    [CLSCompliant(false)]
    public struct OptixDenoiserOptions
    {
        public uint GuideAlbedo;
        public uint GuideNormal;
        public OptixDenoiserAlphaMode DenoiseAlpha;
    }

    public enum OptixPixelFormat
    {
        Half1 = 0x220a,
        Half2 = 0x2207,
        Half3 = 0x2201,
        Half4 = 0x2202,
        Float1 = 0x220b,
        Float2 = 0x2208,
        Float3 = 0x2203,
        Float4 = 0x2204,
        UChar3 = 0x2205,
        UChar4 = 0x2206,
        InternalGuideLayer = 0x2209,
    }

    /// <summary>
    /// Image descriptor used by the denoiser (see optixDenoiserInvoke,
    /// optixDenoiserComputeIntensity).
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixImage2D
    {
        public ulong Data;
        public uint Width;
        public uint Height;
        public uint RowStrideInBytes;
        public uint PixelStrideInBytes;
        public OptixPixelFormat Format;
    }

    [CLSCompliant(false)]
    public struct OptixDenoiserGuideLayer
    {
        public OptixImage2D Albedo;
        public OptixImage2D Normal;
        public OptixImage2D Flow;
        public OptixImage2D PreviousOutputInternalGuideLayer;
        public OptixImage2D OutputInternalGuideLayer;
        public OptixImage2D FlowTrustworthiness;
    }

    public enum OptixDenoiserAOVType
    {
        None = 0,
        Beauty = 0x7000,
        Specular = 0x7001,
        Reflection = 0x7002,
        Refraction = 0x7003,
        Diffuse = 0x7004,
    }

    [CLSCompliant(false)]
    public struct OptixDenoiserLayer
    {
        public OptixImage2D Input;
        public OptixImage2D PreviousOutput;
        public OptixImage2D Output;
        public OptixDenoiserAOVType Type;
    }

    [CLSCompliant(false)]
    public struct OptixDenoiserParams
    {
        /// <summary>
        /// Average log intensity of the input image, or 0 (null) for the denoiser to
        /// compute autoexposure automatically.
        /// </summary>
        public ulong HdrIntensity;

        /// <summary>
        /// 0 = fully denoised output, 1 = unmodified input, values between linearly
        /// interpolate (used for progressive accumulate-then-denoise rendering).
        /// </summary>
        public float BlendFactor;

        /// <summary>
        /// Average log color of the input image (AOV model kind only), or 0 (null)
        /// for the denoiser to compute it automatically.
        /// </summary>
        public ulong HdrAverageColor;

        public uint TemporalModeUsePreviousLayers;
    }

    [CLSCompliant(false)]
    public struct OptixDenoiserSizes
    {
        public ulong StateSizeInBytes;
        public ulong WithOverlapScratchSizeInBytes;
        public ulong WithoutOverlapScratchSizeInBytes;
        public uint OverlapWindowSizeInPixels;
        public ulong ComputeAverageColorSizeInBytes;
        public ulong ComputeIntensitySizeInBytes;
        public ulong InternalGuideLayerPixelSizeInBytes;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1028 // Enum Storage should be Int32
#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1815 // Override equals and operator equals on value types
