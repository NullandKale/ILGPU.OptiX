// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                           Copyright (c) 2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixHitObjectInvoke.tt/OptixHitObjectInvoke.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------


using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of optixInvoke - the third step of the Shader
    /// Execution Reordering (SER) 3-step pattern,
    /// after <see cref="OptixHitObject.Traverse"/> and <see cref="OptixReorder"/>:
    /// actually runs the hit/miss program the traversal found, exchanging payload
    /// registers exactly like <see cref="OptixTrace"/> does. Same "mov.u32 literal
    /// into a scratch register" technique OptixTrace.tt/OptixHitObjectTraverse.tt use
    /// for the payload-type-ID and payload-count arguments - see OptixTrace.cs's own
    /// class doc comment for why a plain EmitRef Input&lt;uint&gt; can't carry them.
    /// </summary>
    public static partial class OptixHitObject
    {
        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying no payload registers.
        /// </summary>
        public static void Invoke(
            )
        {
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %1, 0; mov.u32 %2, 0; call (%0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0), _optix_hitobject_invoke, (%1, %2, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0);"
                , ref _pad
                , ref _type
                , ref _count
                );

        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 1 payload register (p0).
        /// </summary>
        public static void Invoke(
              ref uint p0
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %2, 0; mov.u32 %3, 1; call (%0, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1), _optix_hitobject_invoke, (%2, %3, %0, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1);"
                , ref _p0
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 2 payload registers (p0..p1).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %3, 0; mov.u32 %4, 2; call (%0, %1, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2), _optix_hitobject_invoke, (%3, %4, %0, %1, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2);"
                , ref _p0
                , ref _p1
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 3 payload registers (p0..p2).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %4, 0; mov.u32 %5, 3; call (%0, %1, %2, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3), _optix_hitobject_invoke, (%4, %5, %0, %1, %2, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 4 payload registers (p0..p3).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %5, 0; mov.u32 %6, 4; call (%0, %1, %2, %3, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4), _optix_hitobject_invoke, (%5, %6, %0, %1, %2, %3, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 5 payload registers (p0..p4).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %6, 0; mov.u32 %7, 5; call (%0, %1, %2, %3, %4, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5), _optix_hitobject_invoke, (%6, %7, %0, %1, %2, %3, %4, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 6 payload registers (p0..p5).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %7, 0; mov.u32 %8, 6; call (%0, %1, %2, %3, %4, %5, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6), _optix_hitobject_invoke, (%7, %8, %0, %1, %2, %3, %4, %5, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 7 payload registers (p0..p6).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %8, 0; mov.u32 %9, 7; call (%0, %1, %2, %3, %4, %5, %6, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7), _optix_hitobject_invoke, (%8, %9, %0, %1, %2, %3, %4, %5, %6, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 8 payload registers (p0..p7).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %9, 0; mov.u32 %10, 8; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8), _optix_hitobject_invoke, (%9, %10, %0, %1, %2, %3, %4, %5, %6, %7, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 9 payload registers (p0..p8).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %10, 0; mov.u32 %11, 9; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9), _optix_hitobject_invoke, (%10, %11, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 10 payload registers (p0..p9).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %11, 0; mov.u32 %12, 10; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10), _optix_hitobject_invoke, (%11, %12, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 11 payload registers (p0..p10).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %12, 0; mov.u32 %13, 11; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11), _optix_hitobject_invoke, (%12, %13, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 12 payload registers (p0..p11).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %13, 0; mov.u32 %14, 12; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12), _optix_hitobject_invoke, (%13, %14, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 13 payload registers (p0..p12).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %14, 0; mov.u32 %15, 13; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13), _optix_hitobject_invoke, (%14, %15, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 14 payload registers (p0..p13).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %15, 0; mov.u32 %16, 14; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14), _optix_hitobject_invoke, (%15, %16, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 15 payload registers (p0..p14).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %16, 0; mov.u32 %17, 15; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15), _optix_hitobject_invoke, (%16, %17, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 16 payload registers (p0..p15).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %17, 0; mov.u32 %18, 16; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16), _optix_hitobject_invoke, (%17, %18, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 17 payload registers (p0..p16).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %18, 0; mov.u32 %19, 17; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17), _optix_hitobject_invoke, (%18, %19, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 18 payload registers (p0..p17).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %19, 0; mov.u32 %20, 18; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18), _optix_hitobject_invoke, (%19, %20, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 19 payload registers (p0..p18).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %20, 0; mov.u32 %21, 19; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19), _optix_hitobject_invoke, (%20, %21, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 20 payload registers (p0..p19).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %21, 0; mov.u32 %22, 20; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20), _optix_hitobject_invoke, (%21, %22, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 21 payload registers (p0..p20).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %22, 0; mov.u32 %23, 21; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21), _optix_hitobject_invoke, (%22, %23, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 22 payload registers (p0..p21).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20,
              ref uint p21
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %23, 0; mov.u32 %24, 22; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %22, %22, %22, %22, %22, %22, %22, %22, %22), _optix_hitobject_invoke, (%23, %24, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %22, %22, %22, %22, %22, %22, %22, %22, %22);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 23 payload registers (p0..p22).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20,
              ref uint p21,
              ref uint p22
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %24, 0; mov.u32 %25, 23; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %23, %23, %23, %23, %23, %23, %23, %23), _optix_hitobject_invoke, (%24, %25, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %23, %23, %23, %23, %23, %23, %23, %23);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 24 payload registers (p0..p23).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20,
              ref uint p21,
              ref uint p22,
              ref uint p23
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %25, 0; mov.u32 %26, 24; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %24, %24, %24, %24, %24, %24, %24), _optix_hitobject_invoke, (%25, %26, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %24, %24, %24, %24, %24, %24, %24);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 25 payload registers (p0..p24).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20,
              ref uint p21,
              ref uint p22,
              ref uint p23,
              ref uint p24
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _p24 = p24;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %26, 0; mov.u32 %27, 25; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %25, %25, %25, %25, %25, %25), _optix_hitobject_invoke, (%26, %27, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %25, %25, %25, %25, %25, %25);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _p24
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
            p24 = _p24.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <c>Traverse(...)</c> call, carrying 26 payload registers (p0..p25).
        /// </summary>
        public static void Invoke(
              ref uint p0,
              ref uint p1,
              ref uint p2,
              ref uint p3,
              ref uint p4,
              ref uint p5,
              ref uint p6,
              ref uint p7,
              ref uint p8,
              ref uint p9,
              ref uint p10,
              ref uint p11,
              ref uint p12,
              ref uint p13,
              ref uint p14,
              ref uint p15,
              ref uint p16,
              ref uint p17,
              ref uint p18,
              ref uint p19,
              ref uint p20,
              ref uint p21,
              ref uint p22,
              ref uint p23,
              ref uint p24,
              ref uint p25
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _p24 = p24;
            Ref<uint> _p25 = p25;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;

            CudaAsm.EmitRef(
                "mov.u32 %27, 0; mov.u32 %28, 26; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %26, %26, %26, %26, %26), _optix_hitobject_invoke, (%27, %28, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %26, %26, %26, %26, %26);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _p24
                , ref _p25
                , ref _pad
                , ref _type
                , ref _count
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
            p24 = _p24.Value;
            p25 = _p25.Value;
        }

        /// <summary>
        /// Runs the hit/miss program found by a matching-payload-count
        /// <see cref="Traverse{T}"/> call, carrying a single struct-typed payload
        /// instead of individual <c>ref uint</c> registers - same convention as
        /// <see cref="OptixTrace.Trace{T}"/>, dispatching to one of the fixed-count
        /// <see cref="Invoke"/> overloads above based on <c>sizeof(T)/4</c>.
        /// </summary>
        public static unsafe void Invoke<T>(ref T payload)
            where T : unmanaged
        {
            int count = sizeof(T) / 4;
            T local = payload;
            uint* w = (uint*)&local;

            switch (count)
            {
                case 0:
                    Invoke();
                    break;
                case 1:
                    Invoke(ref w[0]);
                    break;
                case 2:
                    Invoke(ref w[0], ref w[1]);
                    break;
                case 3:
                    Invoke(ref w[0], ref w[1], ref w[2]);
                    break;
                case 4:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3]);
                    break;
                case 5:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4]);
                    break;
                case 6:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5]);
                    break;
                case 7:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6]);
                    break;
                case 8:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7]);
                    break;
                case 9:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8]);
                    break;
                case 10:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9]);
                    break;
                case 11:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10]);
                    break;
                case 12:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11]);
                    break;
                case 13:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12]);
                    break;
                case 14:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13]);
                    break;
                case 15:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14]);
                    break;
                case 16:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15]);
                    break;
                case 17:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16]);
                    break;
                case 18:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17]);
                    break;
                case 19:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18]);
                    break;
                case 20:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19]);
                    break;
                case 21:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20]);
                    break;
                case 22:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20], ref w[21]);
                    break;
                case 23:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20], ref w[21], ref w[22]);
                    break;
                case 24:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20], ref w[21], ref w[22], ref w[23]);
                    break;
                case 25:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20], ref w[21], ref w[22], ref w[23], ref w[24]);
                    break;
                case 26:
                    Invoke(ref w[0], ref w[1], ref w[2], ref w[3], ref w[4], ref w[5], ref w[6], ref w[7], ref w[8], ref w[9], ref w[10], ref w[11], ref w[12], ref w[13], ref w[14], ref w[15], ref w[16], ref w[17], ref w[18], ref w[19], ref w[20], ref w[21], ref w[22], ref w[23], ref w[24], ref w[25]);
                    break;
            }

            payload = local;
        }
    }
}
