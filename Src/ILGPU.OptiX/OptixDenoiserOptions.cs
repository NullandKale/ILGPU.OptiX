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

namespace ILGPU.OptiX
{
    public enum OptixDenoiserModelKind
    {
        OPTIX_DENOISER_MODEL_KIND_AOV = 0x2324,
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV = 0x2326,
        OPTIX_DENOISER_MODEL_KIND_UPSCALE2X = 0x2327,
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X = 0x2328,

        // Deprecated by the SDK - internally mapped to OPTIX_DENOISER_MODEL_KIND_AOV -
        // but this is what the optix7course reference examples (built against an
        // older SDK) use, so it's kept here rather than only exposing the modern name.
        OPTIX_DENOISER_MODEL_KIND_LDR = 0x2322,
        OPTIX_DENOISER_MODEL_KIND_HDR = 0x2323,
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL = 0x2325,
    }

    public enum OptixDenoiserAlphaMode
    {
        OPTIX_DENOISER_ALPHA_MODE_COPY = 0,
        OPTIX_DENOISER_ALPHA_MODE_DENOISE = 1,
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
        OPTIX_PIXEL_FORMAT_HALF1 = 0x220a,
        OPTIX_PIXEL_FORMAT_HALF2 = 0x2207,
        OPTIX_PIXEL_FORMAT_HALF3 = 0x2201,
        OPTIX_PIXEL_FORMAT_HALF4 = 0x2202,
        OPTIX_PIXEL_FORMAT_FLOAT1 = 0x220b,
        OPTIX_PIXEL_FORMAT_FLOAT2 = 0x2208,
        OPTIX_PIXEL_FORMAT_FLOAT3 = 0x2203,
        OPTIX_PIXEL_FORMAT_FLOAT4 = 0x2204,
        OPTIX_PIXEL_FORMAT_UCHAR3 = 0x2205,
        OPTIX_PIXEL_FORMAT_UCHAR4 = 0x2206,
        OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER = 0x2209,
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
        OPTIX_DENOISER_AOV_TYPE_NONE = 0,
        OPTIX_DENOISER_AOV_TYPE_BEAUTY = 0x7000,
        OPTIX_DENOISER_AOV_TYPE_SPECULAR = 0x7001,
        OPTIX_DENOISER_AOV_TYPE_REFLECTION = 0x7002,
        OPTIX_DENOISER_AOV_TYPE_REFRACTION = 0x7003,
        OPTIX_DENOISER_AOV_TYPE_DIFFUSE = 0x7004,
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
