// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixSetPayloadTypes.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixSetPayloadTypes built-in function -
    /// declares which of the module's payload types (see
    /// <see cref="Pipeline.OptixPayloadType"/> and OptixModuleCompileOptions) a
    /// program may be invoked with. Call once at the very start of the program. The
    /// mask must be a compile-time constant per the OptiX ABI - the same PTX
    /// restriction as OptixTrace's payload-type-ID argument - so only baked-literal
    /// declarations are exposed here (bit 0 = payload type 0 =
    /// OPTIX_PAYLOAD_TYPE_ID_0, bit 1 = type 1). Programs that never call this
    /// support all payload types, so this is purely an optimization/validation aid -
    /// the <see cref="OptixTrace.Typed0"/>/<see cref="OptixTrace.Typed1"/> trace
    /// classes work without it.
    /// </summary>
    public static class OptixSetPayloadTypes
    {
        /// <summary>
        /// Declares this program is only invoked with payload type 0
        /// (OPTIX_PAYLOAD_TYPE_ID_0).
        /// </summary>
        public static void DeclareId0() =>
            CudaAsm.Emit(
                "{ .reg .u32 optix_payload_types_mask; " +
                "mov.u32 optix_payload_types_mask, 1; " +
                "call _optix_set_payload_types, (optix_payload_types_mask); }");

        /// <summary>
        /// Declares this program is only invoked with payload type 1
        /// (OPTIX_PAYLOAD_TYPE_ID_1).
        /// </summary>
        public static void DeclareId1() =>
            CudaAsm.Emit(
                "{ .reg .u32 optix_payload_types_mask; " +
                "mov.u32 optix_payload_types_mask, 2; " +
                "call _optix_set_payload_types, (optix_payload_types_mask); }");

        /// <summary>
        /// Declares this program may be invoked with payload type 0 or 1.
        /// </summary>
        public static void DeclareId0AndId1() =>
            CudaAsm.Emit(
                "{ .reg .u32 optix_payload_types_mask; " +
                "mov.u32 optix_payload_types_mask, 3; " +
                "call _optix_set_payload_types, (optix_payload_types_mask); }");
    }
}
