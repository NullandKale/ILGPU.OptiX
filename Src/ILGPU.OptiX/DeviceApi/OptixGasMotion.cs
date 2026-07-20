// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGasMotion.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetGASMotionTimeBegin/TimeEnd/
    /// StepCount built-in functions - the motion options a GAS was built with,
    /// queried by traversable handle (typically <see
    /// cref="OptixGetGASTraversableHandle"/>).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGasMotion
    {
        /// <summary>
        /// The motion time at which the first motion key of the GAS is positioned.
        /// </summary>
        public static float TimeBegin(ulong gasHandle)
        {
            Input<ulong> _handle = gasHandle;
            Output<float> _time = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_gas_motion_time_begin, (%1);",
                ref _time, ref _handle);
            return _time.Value;
        }

        /// <summary>
        /// The motion time at which the last motion key of the GAS is positioned.
        /// </summary>
        public static float TimeEnd(ulong gasHandle)
        {
            Input<ulong> _handle = gasHandle;
            Output<float> _time = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_gas_motion_time_end, (%1);",
                ref _time, ref _handle);
            return _time.Value;
        }

        /// <summary>
        /// The number of motion keys the GAS was built with (0 or 1 when motion is
        /// disabled).
        /// </summary>
        public static uint StepCount(ulong gasHandle)
        {
            Input<ulong> _handle = gasHandle;
            Output<uint> _count = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_gas_motion_step_count, (%1);",
                ref _count, ref _handle);
            return _count.Value;
        }
    }
}
