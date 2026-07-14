// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixBuildInput.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.AccelStructures
{
    public enum OptixBuildInputType
    {
        /// <summary>
        /// Triangle inputs.
        /// </summary>
        Triangles = 0x2141,

        /// <summary>
        /// Custom primitive inputs.
        /// </summary>
        CustomPrimitives = 0x2142,

        /// <summary>
        /// Instance inputs.
        /// </summary>
        Instances = 0x2143,

        /// <summary>
        /// Instance pointer inputs.
        /// </summary>
        InstancePointers = 0x2144,

        /// <summary>
        /// Curve inputs.
        /// </summary>
        Curves = 0x2145
    }

    public enum OptixVertexFormat
    {
        /// <summary>
        /// No vertices
        /// </summary>
        None = 0x0000,

        /// <summary>
        /// Vertices are represented by three floats
        /// </summary>
        Float3 = 0x2121,

        /// <summary>
        /// Vertices are represented by two floats
        /// </summary>
        Float2 = 0x2122,

        /// <summary>
        /// Vertices are represented by three halfs
        /// </summary>
        Half3 = 0x2123,

        /// <summary>
        /// Vertices are represented by two halfs
        /// </summary>
        Half2 = 0x2124,
        Snorm16x3 = 0x2125,
        Snorm16x2 = 0x2126
    }

    public enum OptixIndicesFormat
    {
        /// <summary>
        /// No indices, this format must only be used in combination with triangle soups
        /// i.e., numIndexTriplets must be zero
        /// </summary>
        None = 0,

        /// <summary>
        /// // Three shorts.
        /// </summary>
        UnsignedShort3 = 0x2102,

        /// <summary>
        /// Three ints.
        /// </summary>
        UnsignedInt3 = 0x2103
    }

    public enum OptixTransformFormat
    {
        /// <summary>
        /// No transform, default for zero initialization.
        /// </summary>
        None = 0,

        /// <summary>
        /// 3x4 row major affine matrix.
        /// </summary>
        MatrixFloat12 = 0x21E1,
    }

    public enum OptixPrimitiveType
    {
        /// <summary>
        /// Custom primitive.
        /// </summary>
        Custom = 0x2500,

        /// <summary>
        /// B-spline curve of degree 2 with circular cross-section.
        /// </summary>
        RoundQuadraticBSpline = 0x2501,

        /// <summary>
        /// B-spline curve of degree 3 with circular cross-section.
        /// </summary>
        RoundCubicBSpline = 0x2502,

        /// <summary>
        /// Piecewise linear curve with circular cross-section.
        /// </summary>
        RoundLinear = 0x2503,

        /// <summary>
        /// Triangle.
        /// </summary>
        Triangle = 0x2531,
    }

    /// <summary>
    /// Mirrors OptixCurveEndcapFlags (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [Flags]
    public enum OptixCurveEndcapFlags : uint
    {
        /// <summary>Round end caps for linear, no end caps for quadratic/cubic.</summary>
        Default = 0,

        /// <summary>Flat end caps at both ends of quadratic/cubic curve segments. Not valid for linear.</summary>
        On = 1u << 0,
    }

    [StructLayout(LayoutKind.Explicit)]
    [CLSCompliant(false)]
    public unsafe struct OptixBuildInput
    {
        [FieldOffset(0)]
        public OptixBuildInputType Type;

        [FieldOffset(8)]
        public OptixBuildInputTriangleArray TriangleArray;

        [FieldOffset(8)]
        public OptixBuildInputCurveArray CurveArray;

        [FieldOffset(8)]
        public OptixBuildInputCustomPrimitiveArray CustomPrimitiveArray;

        [FieldOffset(8)]
        public OptixBuildInputInstanceArray InstanceArray;

        [FieldOffset(8)]
        public fixed byte Pad[1024];
    }

    [CLSCompliant(false)]
    public unsafe struct OptixBuildInputTriangleArray
    {
        public IntPtr VertexBuffers;
        public uint NumVertices;
        public OptixVertexFormat VertexFormat;
        public uint VertexStrideInBytes;

        public IntPtr IndexBuffer;
        public uint NumIndexTriplets;
        public OptixIndicesFormat IndexFormat;
        public uint IndexStrideInBytes;

        public IntPtr PreTransform;

        public uint* Flags;

        public uint NumSbtRecords;
        public IntPtr SbtIndexOffsetBuffer;
        public uint SbtIndexOffsetSizeInBytes;
        public uint SbtIndexOffsetStrideInBytes;
        public uint PrimitiveIndexOffset;

        public OptixTransformFormat TransformFormat;

        // This struct previously ended at
        // TransformFormat, omitting OpacityMicromap/DisplacementMicromap entirely (the
        // real SDK 9.0.0 struct has both trailing fields). Only OpacityMicromap is
        // added - displacement micromaps remain out of scope. Safe to omit the trailing DisplacementMicromap field: every
        // OptixBuildInput is constructed via `new OptixBuildInput { ... }`, which
        // zero-initializes the whole union (including this struct's own unused tail
        // bytes) - correctly representing "no displacement micromap" to the driver.
        public OptixBuildInputOpacityMicromap OpacityMicromap;
    }

    // This struct previously omitted IndexBuffer/
    // IndexStrideInBytes/EndcapFlags entirely, which optix_types.h's own doc comment
    // on OptixBuildInputCurveArray::indexBuffer calls out as "required (unlike for
    // OptixBuildInputTriangleArray)" - the struct as it stood could never have
    // actually built curves even with a builder method added, since there was no way
    // to supply the index buffer at all. Fixed to match the real SDK 9.0.0 field
    // order/layout exactly (verified against optix_types.h directly, not guessed).
    [CLSCompliant(false)]
    public unsafe struct OptixBuildInputCurveArray
    {
        public OptixPrimitiveType CurveType;
        public uint NumPrimitives;

        public IntPtr VertexBuffers;
        public uint NumVertices;
        public uint VertexStrideInBytes;

        public IntPtr WidthBuffers;
        public uint WidthStrideInBytes;

        //according to optix_7_types.h this is reserved for future use
        public IntPtr NormalBuffers;
        public uint NormalStrideInBytes;

        /// <summary>
        /// Device pointer to a uint array, one entry per curve segment, each the
        /// index (into VertexBuffers/WidthBuffers) of that segment's first vertex.
        /// Required - unlike triangles, curves have no default/implicit indexing.
        /// </summary>
        public IntPtr IndexBuffer;
        public uint IndexStrideInBytes;

        public uint Flag;

        public uint PrimitiveIndexOffset;

        /// <summary>See <see cref="OptixCurveEndcapFlags"/>.</summary>
        public uint EndcapFlags;
    }

    [CLSCompliant(false)]
    public unsafe struct OptixBuildInputCustomPrimitiveArray
    {
        public IntPtr AabbBuffers;
        public uint NumPrimitives;
        public uint Stride;

        public uint* Flags;

        public uint NumSbtRecords;
        public IntPtr SbtIndexOffsetBuffer;
        public uint SbtIndexOffsetSizeInBytes;
        public uint SbtIndexOffsetStrideInBytes;
        public uint PrimitiveIndexOffset;
    }

    [CLSCompliant(false)]
    public struct OptixBuildInputInstanceArray
    {
        public IntPtr Instances;
        public uint NumInstances;

        /// <summary>
        /// Stride between instances. Zero means tightly packed (stride = sizeof(OptixInstance)).
        /// </summary>
        public uint InstanceStride;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1815 // Override equals and operator equals on value types
