// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Records.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample21
{
    /// <summary>
    /// One per screen pixel that hands off to the NRC cache instead of continuing to
    /// bounce (see <see cref="Core.NrcConstants.FootprintConstant"/> and
    /// RaygenProgram.cs's adaptive ray-footprint handoff test) - written by the raygen
    /// path tracer, read by the composite stage's forward-only cache query. Unlike
    /// <see cref="TrainRecord"/>, there is exactly one slot per pixel (indexed by pixel
    /// index directly), no pooling/counter needed.
    /// </summary>
    public struct EvalRecord
    {
        public byte Active; // 0 = this pixel never handed off (NRC disabled, or path terminated/missed before the handoff bounce) - composite adds nothing.
        public Vec3 Throughput; // Accumulated path throughput up to (and including) the handoff bounce's own BRDF weight.
        public Vec3 Position;
        public Vec3 ScatteredDir;
        public Vec3 Normal;
        public float Roughness;
        public float Metallic;
        public Vec3 Albedo;
    }

    /// <summary>
    /// One per training suffix - a short continuation traced past the eval hand-off
    /// point purely to generate a self-training target for the cache (VkNRC's core
    /// trick: <c>target = bias + factor * cache(endpoint)</c>, resolved by
    /// Device/Network/ForwardKernel-style inference BEFORE the gradient/optimize passes
    /// run. Pooled with an atomic counter
    /// (<see cref="Rendering.TrainRecordBuffers"/>) since only a
    /// <see cref="NrcConstants.TrainProbability"/> fraction of pixels emit one.
    /// </summary>
    public struct TrainRecord
    {
        public Vec3 Bias;   // Radiance accumulated strictly BETWEEN the eval hand-off point and the suffix endpoint (not including the endpoint's own future contribution).
        public Vec3 Factor; // Throughput accumulated over that same suffix (multiplies the endpoint's cache-predicted radiance).
        public Vec3 EndpointPosition;
        public Vec3 EndpointScatteredDir;
        public Vec3 EndpointNormal;
        public float EndpointRoughness;
        public float EndpointMetallic;
        public Vec3 EndpointAlbedo;

        // The record this training STEP actually trains the network against - the
        // eval-hand-off point's own input (NOT the endpoint's) with target = Bias +
        // Factor * cache(endpoint). Kept as separate fields (not reusing Endpoint*)
        // since the gradient pass's forward call needs the START point's encoding while
        // self-training-target resolution needs the END point's.
        public Vec3 StartPosition;
        public Vec3 StartScatteredDir;
        public Vec3 StartNormal;
        public float StartRoughness;
        public float StartMetallic;
        public Vec3 StartAlbedo;

        // Filled by the self-training-target-resolution stage: Bias + Factor *
        // cache(endpoint) - what the gradient pass's loss regresses the network's
        // forward(Start) prediction against.
        public Vec3 Target;
    }
}
