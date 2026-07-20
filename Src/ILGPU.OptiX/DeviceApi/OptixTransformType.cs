// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTransformType.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Mirrors OptixTransformType (optix_types.h) - the kind of traversable a
    /// transform-list handle refers to, as returned by
    /// <see cref="OptixTransformList.GetTransformType"/>.
    /// </summary>
    public enum OptixTransformType
    {
        /// <summary>
        /// Not a transformation.
        /// </summary>
        None = 0,

        /// <summary>
        /// An OptixStaticTransform traversable.
        /// </summary>
        StaticTransform = 1,

        /// <summary>
        /// An OptixMatrixMotionTransform traversable.
        /// </summary>
        MatrixMotionTransform = 2,

        /// <summary>
        /// An OptixSRTMotionTransform traversable.
        /// </summary>
        SrtMotionTransform = 3,

        /// <summary>
        /// An instance within an IAS (see OptixInstance).
        /// </summary>
        Instance = 4,
    }
}
