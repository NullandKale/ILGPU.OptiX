// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTransforms.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Accessor over the hit-state-specific intrinsics the transform composition in
    /// <see cref="OptixTransforms"/> needs. The SDK expresses the same abstraction as
    /// the HitState template parameter in optix_device_impl.h: the current-hit
    /// (IS/AH/CH) and hit-object (SER) program states expose the same transform
    /// stack through differently-named pseudo-calls, and generic specialization
    /// keeps each program's PTX free of the other state's intrinsics.
    /// </summary>
    internal interface IOptixHitStateAccessor
    {
        uint TransformListSize { get; }
        ulong GetTransformListHandle(uint index);
        float RayTime { get; }
    }

    /// <summary>
    /// The current-hit (IS/AH/CH) transform-stack accessor, backed by
    /// _optix_get_transform_list_* and _optix_get_ray_time.
    /// </summary>
    internal readonly struct OptixCurrentHitAccessor : IOptixHitStateAccessor
    {
        public uint TransformListSize => OptixTransformList.Size;

        public ulong GetTransformListHandle(uint index) =>
            OptixTransformList.GetHandle(index);

        public float RayTime => OptixGetRayTime.Value;
    }

    /// <summary>
    /// Provides the functionality of the transformation helper built-in functions
    /// (optixGetWorldToObjectTransformMatrix, optixGetObjectToWorldTransformMatrix,
    /// and the six optixTransform{Point,Vector,Normal}From{World,Object}To*Space
    /// functions). Valid in IS/AH/CH programs.
    /// </summary>
    /// <remarks>
    /// There are no matrix intrinsics in OptiX - the SDK composes these matrices in
    /// C++ (optix_device_impl_transformations.h) from the raw transform-list
    /// intrinsics exposed here via <see cref="OptixTransformList"/>. This class is a
    /// faithful port of that composition, including motion-key interpolation for
    /// matrix- and SRT-motion transforms; keep any behavioral change in sync with
    /// the SDK header.
    /// </remarks>
    public static class OptixTransforms
    {
        // -------------------------------------------------------------------------
        // 16-byte-aligned global loads (port of optixLdg / optixLoadReadOnlyAlign16)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Loads a float4 from a 16-byte-aligned device address via a global-space
        /// vector load (port of the SDK's optixLdg). The transform structs OptiX
        /// hands out via <see cref="OptixTransformList"/> guarantee this alignment.
        /// </summary>
        [CLSCompliant(false)]
        public static Vector4 LoadFloat4(ulong address)
        {
            Input<ulong> _address = address;
            Output<ulong> _global = default;
            CudaAsm.EmitRef(
                "cvta.to.global.u64 %0, %1;",
                ref _global, ref _address);

            Input<ulong> _pointer = _global.Value;
            Output<uint> _x = default;
            Output<uint> _y = default;
            Output<uint> _z = default;
            Output<uint> _w = default;
            CudaAsm.EmitRef(
                "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
                ref _x, ref _y, ref _z, ref _w, ref _pointer);

            return new Vector4(
                ILGPU.Interop.IntAsFloat(_x.Value),
                ILGPU.Interop.IntAsFloat(_y.Value),
                ILGPU.Interop.IntAsFloat(_z.Value),
                ILGPU.Interop.IntAsFloat(_w.Value));
        }

        [CLSCompliant(false)]
        public static (uint X, uint Y, uint Z, uint W) LoadUInt4(ulong address)
        {
            Input<ulong> _address = address;
            Output<ulong> _global = default;
            CudaAsm.EmitRef(
                "cvta.to.global.u64 %0, %1;",
                ref _global, ref _address);

            Input<ulong> _pointer = _global.Value;
            Output<uint> _x = default;
            Output<uint> _y = default;
            Output<uint> _z = default;
            Output<uint> _w = default;
            CudaAsm.EmitRef(
                "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
                ref _x, ref _y, ref _z, ref _w, ref _pointer);

            return (_x.Value, _y.Value, _z.Value, _w.Value);
        }

        // -------------------------------------------------------------------------
        // Struct layout constants (optix_types.h; all structs are 16-byte aligned)
        // -------------------------------------------------------------------------

        // OptixMatrixMotionTransform / OptixSRTMotionTransform share the same
        // header: child handle (8 bytes), OptixMotionOptions (numKeys u16, flags
        // u16, timeBegin f32, timeEnd f32 = 12 bytes), pad to 16-byte alignment.
        // The key data starts at byte 32 in both.
        private const uint MotionTransformKeysOffset = 32;
        private const uint MatrixKeySizeInBytes = 48;  // 12 floats
        private const uint SrtKeySizeInBytes = 64;     // 16 floats

        // OptixStaticTransform: child handle (8) + pad[2] (8), then transform[12]
        // at byte 16 and invTransform[12] at byte 64.
        private const uint StaticTransformOffset = 16;
        private const uint StaticInverseTransformOffset = 64;

        // -------------------------------------------------------------------------
        // Matrix math (ports of the optix_impl helpers)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Multiplies the row vector with the 3x4 matrix given by rows m0, m1, m2
        /// (port of optixMultiplyRowMatrix).
        /// </summary>
        public static Vector4 MultiplyRowMatrix(
            Vector4 vec,
            Vector4 m0,
            Vector4 m1,
            Vector4 m2) =>
            new Vector4(
                vec.X * m0.X + vec.Y * m1.X + vec.Z * m2.X,
                vec.X * m0.Y + vec.Y * m1.Y + vec.Z * m2.Y,
                vec.X * m0.Z + vec.Y * m1.Z + vec.Z * m2.Z,
                vec.X * m0.W + vec.Y * m1.W + vec.Z * m2.W + vec.W);

        /// <summary>
        /// Converts an SRT transformation (the four rows of an OptixSRTData: sx a b
        /// pvx | sy c pvy sz | pvz qx qy qz | qw tx ty tz) into a 3x4 matrix (port
        /// of optixGetMatrixFromSrt).
        /// </summary>
        public static void GetMatrixFromSrt(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2,
            Vector4 srt0,
            Vector4 srt1,
            Vector4 srt2,
            Vector4 srt3)
        {
            float sx = srt0.X, a = srt0.Y, b = srt0.Z, pvx = srt0.W;
            float sy = srt1.X, c = srt1.Y, pvy = srt1.Z, sz = srt1.W;
            float pvz = srt2.X, qx = srt2.Y, qy = srt2.Z, qz = srt2.W;
            float qw = srt3.X, tx = srt3.Y, ty = srt3.Z, tz = srt3.W;

            // The quaternion is assumed to be normalized.
            float sqw = qw * qw;
            float sqx = qx * qx;
            float sqy = qy * qy;
            float sqz = qz * qz;

            float xy = qx * qy;
            float zw = qz * qw;
            float xz = qx * qz;
            float yw = qy * qw;
            float yz = qy * qz;
            float xw = qx * qw;

            m0 = default;
            m1 = default;
            m2 = default;

            m0.X = sqx - sqy - sqz + sqw;
            m0.Y = 2.0f * (xy - zw);
            m0.Z = 2.0f * (xz + yw);

            m1.X = 2.0f * (xy + zw);
            m1.Y = -sqx + sqy - sqz + sqw;
            m1.Z = 2.0f * (yz - xw);

            m2.X = 2.0f * (xz - yw);
            m2.Y = 2.0f * (yz + xw);
            m2.Z = -sqx - sqy + sqz + sqw;

            m0.W = m0.X * pvx + m0.Y * pvy + m0.Z * pvz + tx;
            m1.W = m1.X * pvx + m1.Y * pvy + m1.Z * pvz + ty;
            m2.W = m2.X * pvx + m2.Y * pvy + m2.Z * pvz + tz;

            m0.Z = m0.X * b + m0.Y * c + m0.Z * sz;
            m1.Z = m1.X * b + m1.Y * c + m1.Z * sz;
            m2.Z = m2.X * b + m2.Y * c + m2.Z * sz;

            m0.Y = m0.X * a + m0.Y * sy;
            m1.Y = m1.X * a + m1.Y * sy;
            m2.Y = m2.X * a + m2.Y * sy;

            m0.X *= sx;
            m1.X *= sx;
            m2.X *= sx;
        }

        /// <summary>
        /// Inverts a 3x4 affine matrix in place (port of optixInvertMatrix).
        /// </summary>
        public static void InvertMatrix(
            ref Vector4 m0,
            ref Vector4 m1,
            ref Vector4 m2)
        {
            float det3 =
                m0.X * (m1.Y * m2.Z - m1.Z * m2.Y) -
                m0.Y * (m1.X * m2.Z - m1.Z * m2.X) +
                m0.Z * (m1.X * m2.Y - m1.Y * m2.X);

            float invDet3 = 1.0f / det3;

            float i00 = invDet3 * (m1.Y * m2.Z - m2.Y * m1.Z);
            float i01 = invDet3 * (m0.Z * m2.Y - m2.Z * m0.Y);
            float i02 = invDet3 * (m0.Y * m1.Z - m1.Y * m0.Z);

            float i10 = invDet3 * (m1.Z * m2.X - m2.Z * m1.X);
            float i11 = invDet3 * (m0.X * m2.Z - m2.X * m0.Z);
            float i12 = invDet3 * (m0.Z * m1.X - m1.Z * m0.X);

            float i20 = invDet3 * (m1.X * m2.Y - m2.X * m1.Y);
            float i21 = invDet3 * (m0.Y * m2.X - m2.Y * m0.X);
            float i22 = invDet3 * (m0.X * m1.Y - m1.X * m0.Y);

            float b0 = m0.W;
            float b1 = m1.W;
            float b2 = m2.W;

            m0 = new Vector4(i00, i01, i02, -i00 * b0 - i01 * b1 - i02 * b2);
            m1 = new Vector4(i10, i11, i12, -i10 * b0 - i11 * b1 - i12 * b2);
            m2 = new Vector4(i20, i21, i22, -i20 * b0 - i21 * b1 - i22 * b2);
        }

        // -------------------------------------------------------------------------
        // Motion key resolution and interpolation
        // -------------------------------------------------------------------------

        private static void ResolveMotionKey(
            out float localTime,
            out int key,
            uint numKeys,
            float timeBegin,
            float timeEnd,
            float globalTime)
        {
            float numIntervals = numKeys - 1;

            // Should be NaN or in [0, numIntervals].
            float time = Math.Max(
                0.0f,
                Math.Min(
                    numIntervals,
                    numIntervals * ((globalTime - timeBegin) / (timeEnd - timeBegin))));

            // Catch NaN (for example when timeBegin == timeEnd).
            if (float.IsNaN(time))
                time = 0.0f;

            float fltKey = Math.Min((float)Math.Floor(time), numIntervals - 1);

            localTime = time - fltKey;
            key = (int)fltKey;
        }

        private static void LoadInterpolatedMatrixKey(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2,
            ulong keyAddress,
            float t1)
        {
            m0 = LoadFloat4(keyAddress);
            m1 = LoadFloat4(keyAddress + 16);
            m2 = LoadFloat4(keyAddress + 32);

            // The next key's rows follow contiguously (12 floats per key). The
            // conditional matches the SDK: it prevents concurrent loads leading to
            // spills.
            if (t1 > 0.0f)
            {
                float t0 = 1.0f - t1;
                m0 = m0 * t0 + LoadFloat4(keyAddress + 48) * t1;
                m1 = m1 * t0 + LoadFloat4(keyAddress + 64) * t1;
                m2 = m2 * t0 + LoadFloat4(keyAddress + 80) * t1;
            }
        }

        private static void LoadInterpolatedSrtKey(
            out Vector4 srt0,
            out Vector4 srt1,
            out Vector4 srt2,
            out Vector4 srt3,
            ulong keyAddress,
            float t1)
        {
            srt0 = LoadFloat4(keyAddress);
            srt1 = LoadFloat4(keyAddress + 16);
            srt2 = LoadFloat4(keyAddress + 32);
            srt3 = LoadFloat4(keyAddress + 48);

            // The next key's rows follow contiguously (16 floats per key).
            if (t1 > 0.0f)
            {
                float t0 = 1.0f - t1;
                srt0 = srt0 * t0 + LoadFloat4(keyAddress + 64) * t1;
                srt1 = srt1 * t0 + LoadFloat4(keyAddress + 80) * t1;
                srt2 = srt2 * t0 + LoadFloat4(keyAddress + 96) * t1;
                srt3 = srt3 * t0 + LoadFloat4(keyAddress + 112) * t1;

                // Renormalize the interpolated quaternion (qx qy qz qw live in
                // srt2.YZW and srt3.X).
                float invLength = 1.0f / (float)Math.Sqrt(
                    srt2.Y * srt2.Y + srt2.Z * srt2.Z + srt2.W * srt2.W +
                    srt3.X * srt3.X);
                srt2.Y *= invLength;
                srt2.Z *= invLength;
                srt2.W *= invLength;
                srt3.X *= invLength;
            }
        }

        /// <summary>
        /// Reads the OptixMotionOptions header shared by both motion transform
        /// structs: numKeys/flags/timeBegin sit in the first 16 bytes, timeEnd in
        /// the second 16-byte block.
        /// </summary>
        private static void LoadMotionOptions(
            ulong transformAddress,
            out uint numKeys,
            out float timeBegin,
            out float timeEnd)
        {
            var (_, _, packedKeys, timeBeginBits) = LoadUInt4(transformAddress);
            var (timeEndBits, _, _, _) = LoadUInt4(transformAddress + 16);
            numKeys = packedKeys & 0xFFFF;
            timeBegin = ILGPU.Interop.IntAsFloat(timeBeginBits);
            timeEnd = ILGPU.Interop.IntAsFloat(timeEndBits);
        }

        /// <summary>
        /// Returns the interpolated 3x4 transformation matrix for a transform-list
        /// handle and point in time (port of
        /// optixGetInterpolatedTransformationFromHandle).
        /// </summary>
        [CLSCompliant(false)]
        public static void GetInterpolatedTransformationFromHandle(
            out Vector4 trf0,
            out Vector4 trf1,
            out Vector4 trf2,
            ulong handle,
            float time,
            bool objectToWorld)
        {
            var type = OptixTransformList.GetTransformType(handle);

            if (type == OptixTransformType.MatrixMotionTransform ||
                type == OptixTransformType.SrtMotionTransform)
            {
                if (type == OptixTransformType.MatrixMotionTransform)
                {
                    ulong data =
                        OptixTransformList.GetMatrixMotionTransformPointer(handle);
                    LoadMotionOptions(
                        data,
                        out uint numKeys,
                        out float timeBegin,
                        out float timeEnd);
                    ResolveMotionKey(
                        out float keyTime,
                        out int key,
                        numKeys,
                        timeBegin,
                        timeEnd,
                        time);
                    LoadInterpolatedMatrixKey(
                        out trf0,
                        out trf1,
                        out trf2,
                        data + MotionTransformKeysOffset +
                            (uint)key * MatrixKeySizeInBytes,
                        keyTime);
                }
                else
                {
                    ulong data =
                        OptixTransformList.GetSrtMotionTransformPointer(handle);
                    LoadMotionOptions(
                        data,
                        out uint numKeys,
                        out float timeBegin,
                        out float timeEnd);
                    ResolveMotionKey(
                        out float keyTime,
                        out int key,
                        numKeys,
                        timeBegin,
                        timeEnd,
                        time);
                    LoadInterpolatedSrtKey(
                        out Vector4 srt0,
                        out Vector4 srt1,
                        out Vector4 srt2,
                        out Vector4 srt3,
                        data + MotionTransformKeysOffset +
                            (uint)key * SrtKeySizeInBytes,
                        keyTime);
                    GetMatrixFromSrt(
                        out trf0,
                        out trf1,
                        out trf2,
                        srt0,
                        srt1,
                        srt2,
                        srt3);
                }

                if (!objectToWorld)
                    InvertMatrix(ref trf0, ref trf1, ref trf2);
            }
            else if (type == OptixTransformType.Instance ||
                type == OptixTransformType.StaticTransform)
            {
                ulong transform;
                if (type == OptixTransformType.Instance)
                {
                    transform = objectToWorld
                        ? OptixTransformList.GetInstanceTransformPointer(handle)
                        : OptixTransformList.GetInstanceInverseTransformPointer(
                            handle);
                }
                else
                {
                    ulong traversable =
                        OptixTransformList.GetStaticTransformPointer(handle);
                    transform = traversable + (objectToWorld
                        ? StaticTransformOffset
                        : StaticInverseTransformOffset);
                }

                trf0 = LoadFloat4(transform);
                trf1 = LoadFloat4(transform + 16);
                trf2 = LoadFloat4(transform + 32);
            }
            else
            {
                trf0 = new Vector4(1.0f, 0.0f, 0.0f, 0.0f);
                trf1 = new Vector4(0.0f, 1.0f, 0.0f, 0.0f);
                trf2 = new Vector4(0.0f, 0.0f, 1.0f, 0.0f);
            }
        }

        // -------------------------------------------------------------------------
        // Transform-stack composition (generic over the hit state; see
        // IOptixHitStateAccessor)
        // -------------------------------------------------------------------------

        internal static void GetWorldToObjectTransformMatrix<THitState>(
            THitState hitState,
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2)
            where THitState : struct, IOptixHitStateAccessor
        {
            m0 = new Vector4(1.0f, 0.0f, 0.0f, 0.0f);
            m1 = new Vector4(0.0f, 1.0f, 0.0f, 0.0f);
            m2 = new Vector4(0.0f, 0.0f, 1.0f, 0.0f);

            uint size = hitState.TransformListSize;
            float time = hitState.RayTime;

            for (uint i = 0; i < size; ++i)
            {
                ulong handle = hitState.GetTransformListHandle(i);

                GetInterpolatedTransformationFromHandle(
                    out Vector4 trf0,
                    out Vector4 trf1,
                    out Vector4 trf2,
                    handle,
                    time,
                    objectToWorld: false);

                if (i == 0)
                {
                    m0 = trf0;
                    m1 = trf1;
                    m2 = trf2;
                }
                else
                {
                    // m := trf * m
                    Vector4 tmp0 = m0, tmp1 = m1, tmp2 = m2;
                    m0 = MultiplyRowMatrix(trf0, tmp0, tmp1, tmp2);
                    m1 = MultiplyRowMatrix(trf1, tmp0, tmp1, tmp2);
                    m2 = MultiplyRowMatrix(trf2, tmp0, tmp1, tmp2);
                }
            }
        }

        internal static void GetObjectToWorldTransformMatrix<THitState>(
            THitState hitState,
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2)
            where THitState : struct, IOptixHitStateAccessor
        {
            m0 = new Vector4(1.0f, 0.0f, 0.0f, 0.0f);
            m1 = new Vector4(0.0f, 1.0f, 0.0f, 0.0f);
            m2 = new Vector4(0.0f, 0.0f, 1.0f, 0.0f);

            int size = (int)hitState.TransformListSize;
            float time = hitState.RayTime;

            for (int i = size - 1; i >= 0; --i)
            {
                ulong handle = hitState.GetTransformListHandle((uint)i);

                GetInterpolatedTransformationFromHandle(
                    out Vector4 trf0,
                    out Vector4 trf1,
                    out Vector4 trf2,
                    handle,
                    time,
                    objectToWorld: true);

                if (i == size - 1)
                {
                    m0 = trf0;
                    m1 = trf1;
                    m2 = trf2;
                }
                else
                {
                    // m := trf * m
                    Vector4 tmp0 = m0, tmp1 = m1, tmp2 = m2;
                    m0 = MultiplyRowMatrix(trf0, tmp0, tmp1, tmp2);
                    m1 = MultiplyRowMatrix(trf1, tmp0, tmp1, tmp2);
                    m2 = MultiplyRowMatrix(trf2, tmp0, tmp1, tmp2);
                }
            }
        }

        // -------------------------------------------------------------------------
        // 3x4 matrix application (ports of optixTransformPoint/Vector/Normal)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Multiplies the 3x4 matrix given by rows m0, m1, m2 with the point (port
        /// of optixTransformPoint).
        /// </summary>
        public static Vector3 TransformPoint(
            Vector4 m0,
            Vector4 m1,
            Vector4 m2,
            Vector3 point) =>
            new Vector3(
                m0.X * point.X + m0.Y * point.Y + m0.Z * point.Z + m0.W,
                m1.X * point.X + m1.Y * point.Y + m1.Z * point.Z + m1.W,
                m2.X * point.X + m2.Y * point.Y + m2.Z * point.Z + m2.W);

        /// <summary>
        /// Multiplies the 3x3 linear submatrix of the 3x4 matrix with the vector
        /// (port of optixTransformVector).
        /// </summary>
        public static Vector3 TransformVector(
            Vector4 m0,
            Vector4 m1,
            Vector4 m2,
            Vector3 vector) =>
            new Vector3(
                m0.X * vector.X + m0.Y * vector.Y + m0.Z * vector.Z,
                m1.X * vector.X + m1.Y * vector.Y + m1.Z * vector.Z,
                m2.X * vector.X + m2.Y * vector.Y + m2.Z * vector.Z);

        /// <summary>
        /// Multiplies the transpose of the 3x3 linear submatrix with the normal
        /// (port of optixTransformNormal). The matrix passed in must be the inverse
        /// of the actual transformation, per the SDK contract.
        /// </summary>
        public static Vector3 TransformNormal(
            Vector4 m0,
            Vector4 m1,
            Vector4 m2,
            Vector3 normal) =>
            new Vector3(
                m0.X * normal.X + m1.X * normal.Y + m2.X * normal.Z,
                m0.Y * normal.X + m1.Y * normal.Y + m2.Y * normal.Z,
                m0.Z * normal.X + m1.Z * normal.Y + m2.Z * normal.Z);

        // -------------------------------------------------------------------------
        // Public current-hit surface (ports of the optix_device.h entry points)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Provides the functionality of optixGetWorldToObjectTransformMatrix - the
        /// composed world-to-object 3x4 matrix of the current hit's transform stack
        /// (identity when the stack is empty).
        /// </summary>
        public static void GetWorldToObjectTransformMatrix(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2) =>
            GetWorldToObjectTransformMatrix(
                default(OptixCurrentHitAccessor),
                out m0,
                out m1,
                out m2);

        /// <summary>
        /// Provides the functionality of optixGetObjectToWorldTransformMatrix - the
        /// composed object-to-world 3x4 matrix of the current hit's transform stack
        /// (identity when the stack is empty).
        /// </summary>
        public static void GetObjectToWorldTransformMatrix(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2) =>
            GetObjectToWorldTransformMatrix(
                default(OptixCurrentHitAccessor),
                out m0,
                out m1,
                out m2);

        /// <summary>
        /// Provides the functionality of optixTransformPointFromWorldToObjectSpace.
        /// </summary>
        public static Vector3 TransformPointFromWorldToObjectSpace(Vector3 point)
        {
            if (OptixTransformList.Size == 0)
                return point;

            GetWorldToObjectTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformPoint(m0, m1, m2, point);
        }

        /// <summary>
        /// Provides the functionality of optixTransformVectorFromWorldToObjectSpace.
        /// </summary>
        public static Vector3 TransformVectorFromWorldToObjectSpace(Vector3 vector)
        {
            if (OptixTransformList.Size == 0)
                return vector;

            GetWorldToObjectTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformVector(m0, m1, m2, vector);
        }

        /// <summary>
        /// Provides the functionality of optixTransformNormalFromWorldToObjectSpace.
        /// Note the SDK contract: the normal is multiplied with the transpose of the
        /// object-to-world matrix (the inverse of world-to-object).
        /// </summary>
        public static Vector3 TransformNormalFromWorldToObjectSpace(Vector3 normal)
        {
            if (OptixTransformList.Size == 0)
                return normal;

            GetObjectToWorldTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformNormal(m0, m1, m2, normal);
        }

        /// <summary>
        /// Provides the functionality of optixTransformPointFromObjectToWorldSpace.
        /// </summary>
        public static Vector3 TransformPointFromObjectToWorldSpace(Vector3 point)
        {
            if (OptixTransformList.Size == 0)
                return point;

            GetObjectToWorldTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformPoint(m0, m1, m2, point);
        }

        /// <summary>
        /// Provides the functionality of optixTransformVectorFromObjectToWorldSpace.
        /// </summary>
        public static Vector3 TransformVectorFromObjectToWorldSpace(Vector3 vector)
        {
            if (OptixTransformList.Size == 0)
                return vector;

            GetObjectToWorldTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformVector(m0, m1, m2, vector);
        }

        /// <summary>
        /// Provides the functionality of optixTransformNormalFromObjectToWorldSpace.
        /// Note the SDK contract: the normal is multiplied with the transpose of the
        /// world-to-object matrix (the inverse of object-to-world).
        /// </summary>
        public static Vector3 TransformNormalFromObjectToWorldSpace(Vector3 normal)
        {
            if (OptixTransformList.Size == 0)
                return normal;

            GetWorldToObjectTransformMatrix(
                out Vector4 m0,
                out Vector4 m1,
                out Vector4 m2);
            return TransformNormal(m0, m1, m2, normal);
        }
    }
}
