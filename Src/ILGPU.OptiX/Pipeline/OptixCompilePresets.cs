// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCompilePresets.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Static presets for module compile options.
    /// </summary>
    public static class OptixCompilePresets
    {
        /// <summary>
        /// Debug preset: OptimizationLevel = Level0, DebugLevel = Full.
        /// </summary>
        public static readonly OptixModuleCompileOptions Debug = new OptixModuleCompileOptions
        {
            OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_FULL,
        };

        /// <summary>
        /// Release preset: OptimizationLevel = Default, DebugLevel = None.
        /// </summary>
        public static readonly OptixModuleCompileOptions Release = new OptixModuleCompileOptions
        {
            OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        };
    }
}
