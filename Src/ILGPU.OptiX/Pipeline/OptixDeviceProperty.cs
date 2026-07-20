// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDeviceProperty.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Mirrors OptixDeviceProperty (optix_types.h) - queryable via
    /// <see cref="OptixDeviceContextExtensions.GetProperty"/>. Every property is a
    /// 4-byte unsigned value.
    /// </summary>
    public enum OptixDeviceProperty
    {
        /// <summary>Maximum value for OptixPipelineLinkOptions.MaxTraceDepth.</summary>
        LimitMaxTraceDepth = 0x2001,

        /// <summary>Maximum maxTraversableGraphDepth for optixPipelineSetStackSize.</summary>
        LimitMaxTraversableGraphDepth = 0x2002,

        /// <summary>Maximum number of primitives (over all build inputs) per GAS.</summary>
        LimitMaxPrimitivesPerGas = 0x2003,

        /// <summary>Maximum number of instances (over all build inputs) per IAS.</summary>
        LimitMaxInstancesPerIas = 0x2004,

        /// <summary>
        /// The RT core version supported by the device (0 for no support, 10 for
        /// version 1.0).
        /// </summary>
        RtcoreVersion = 0x2005,

        /// <summary>Maximum value for OptixInstance.InstanceId.</summary>
        LimitMaxInstanceId = 0x2006,

        /// <summary>Number of bits available for OptixInstance.VisibilityMask.</summary>
        LimitNumBitsInstanceVisibilityMask = 0x2007,

        /// <summary>Maximum number of SBT records per GAS.</summary>
        LimitMaxSbtRecordsPerGas = 0x2008,

        /// <summary>Maximum summed value of OptixInstance.SbtOffset.</summary>
        LimitMaxSbtOffset = 0x2009,

        /// <summary>
        /// Capabilities of optixReorder (0 = none, bit 0 = standard SER support).
        /// </summary>
        ShaderExecutionReordering = 0x200A,

        /// <summary>
        /// Cooperative vector support (0 = none, bit 0 = standard support).
        /// </summary>
        CoopVec = 0x200B,

        /// <summary>
        /// Cluster acceleration structure support (0 = none, bit 0 = standard
        /// support).
        /// </summary>
        ClusterAccel = 0x2020,

        /// <summary>Maximum number of unique vertices per cluster.</summary>
        LimitMaxClusterVertices = 0x2021,

        /// <summary>Maximum number of triangles per cluster.</summary>
        LimitMaxClusterTriangles = 0x2022,

        /// <summary>Maximum resolution of a structured-grid cluster template.</summary>
        LimitMaxStructuredGridResolution = 0x2023,
    }
}
