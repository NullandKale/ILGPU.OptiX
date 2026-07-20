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
        Curves = 0x2145,

        /// <summary>
        /// Sphere inputs.
        /// </summary>
        Spheres = 0x2146
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
        /// CatmullRom curve with circular cross-section.
        /// </summary>
        RoundCatmullRom = 0x2504,

        /// <summary>
        /// B-spline curve of degree 2 with oriented flat cross-section (ribbon).
        /// </summary>
        FlatQuadraticBSpline = 0x2505,

        /// <summary>
        /// Sphere.
        /// </summary>
        Sphere = 0x2506,

        /// <summary>
        /// Bezier curve of degree 3 with circular cross-section.
        /// </summary>
        RoundCubicBezier = 0x2507,

        /// <summary>
        /// B-spline curve of degree 2 with circular cross-section,
        /// with rounded endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundQuadraticBSplineRocaps = 0x2508,

        /// <summary>
        /// B-spline curve of degree 3 with circular cross-section,
        /// with rounded endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundCubicBSplineRocaps = 0x2509,

        /// <summary>
        /// CatmullRom curve with circular cross-section,
        /// with rounded endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundCatmullRomRocaps = 0x250A,

        /// <summary>
        /// Bezier curve of degree 3 with circular cross-section,
        /// with rounded endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundCubicBezierRocaps = 0x250B,

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
        public OptixBuildInputSphereArray SphereArray;

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

        // DisplacementMicromap (the SDK struct's other trailing field) is intentionally
        // omitted - out of scope. Safe: every OptixBuildInput is constructed via
        // `new OptixBuildInput { ... }`, which zero-initializes the whole union, so the
        // omitted tail bytes correctly read as "no displacement micromap" to the driver.
        public OptixBuildInputOpacityMicromap OpacityMicromap;
    }

    // Mirrors OptixBuildInputCurveArray (optix_types.h). IndexBuffer is required per
    // the SDK's own doc comment, unlike OptixBuildInputTriangleArray.
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

    /// <summary>
    /// Mirrors OptixBuildInputSphereArray (optix_types.h) - built-in sphere
    /// primitives, one sphere per center vertex. Requires the pipeline to be
    /// compiled with <see cref="OptixPrimitiveTypeFlags.Sphere"/> and hit groups
    /// using the sphere builtin intersection module.
    /// </summary>
    [CLSCompliant(false)]
    public unsafe struct OptixBuildInputSphereArray
    {
        /// <summary>
        /// Pointer to a host array of device pointers (one per motion step), each
        /// pointing to an array of float3 sphere center points.
        /// </summary>
        public IntPtr VertexBuffers;

        /// <summary>Stride between vertices; zero means tightly packed float3.</summary>
        public uint VertexStrideInBytes;

        /// <summary>Number of vertices in each buffer in <see cref="VertexBuffers"/>.</summary>
        public uint NumVertices;

        /// <summary>
        /// Parallel to <see cref="VertexBuffers"/>: a device pointer per motion
        /// step, each with per-vertex float radii (or a single float when
        /// <see cref="SingleRadius"/> is set).
        /// </summary>
        public IntPtr RadiusBuffers;

        /// <summary>Stride between radii; zero means tightly packed floats.</summary>
        public uint RadiusStrideInBytes;

        /// <summary>
        /// Boolean - when non-zero each radius buffer holds a single radius shared
        /// by all spheres instead of one per vertex.
        /// </summary>
        public int SingleRadius;

        /// <summary>
        /// Host array of per-SBT-record OptixGeometryFlags; size must match
        /// <see cref="NumSbtRecords"/>.
        /// </summary>
        public uint* Flags;

        /// <summary>Number of SBT records available to the SBT index offset override.</summary>
        public uint NumSbtRecords;

        /// <summary>
        /// Device pointer to a per-primitive local SBT index offset buffer; may be
        /// null. Every entry must be in [0, NumSbtRecords-1].
        /// </summary>
        public IntPtr SbtIndexOffsetBuffer;

        /// <summary>Size of the SBT index offset type: 0, 1, 2 or 4 bytes.</summary>
        public uint SbtIndexOffsetSizeInBytes;

        /// <summary>Stride between SBT index offsets; zero means tightly packed.</summary>
        public uint SbtIndexOffsetStrideInBytes;

        /// <summary>Primitive index bias applied in optixGetPrimitiveIndex().</summary>
        public uint PrimitiveIndexOffset;
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
