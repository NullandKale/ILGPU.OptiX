// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPayloadDefaults.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.Device
{
    /// <summary>
    /// Standard ray-type and bounce-flag constants used across path-tracing samples.
    /// Provides a common vocabulary and eliminates duplicate constant definitions.
    /// Samples can extend this with their own payload layout helpers as needed.
    /// </summary>
    public static class OptixPayloadDefaults
    {
        /// <summary>
        /// Ray type for primary/indirect radiance rays.
        /// </summary>
        public const uint RADIANCE_RAY_TYPE = 0;

        /// <summary>
        /// Ray type for shadow/occlusion rays.
        /// </summary>
        public const uint SHADOW_RAY_TYPE = 1;

        /// <summary>
        /// Total number of ray types used in standard path tracing.
        /// </summary>
        public const uint RAY_TYPE_COUNT = 2;

        /// <summary>
        /// Continuation flag: terminal bounce (no further tracing needed).
        /// </summary>
        public const uint BOUNCE_TERMINAL = 0;

        /// <summary>
        /// Continuation flag: continue with mirror/specular reflection.
        /// </summary>
        public const uint BOUNCE_CONTINUE_MIRROR = 1;

        /// <summary>
        /// Continuation flag: continue with dielectric (refraction/transmission).
        /// </summary>
        public const uint BOUNCE_CONTINUE_DIELECTRIC = 2;

        /// <summary>
        /// Continuation flag: continue with diffuse/Lambertian reflection.
        /// </summary>
        public const uint BOUNCE_CONTINUE_DIFFUSE = 3;
    }
}
