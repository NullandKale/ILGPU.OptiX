// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: SelfTrainingTargetProgram.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.DeviceApi;
using Sample21.Device.Network;
using Half = ILGPU.Half;

namespace Sample21.Device
{
    /// <summary>
    /// NRC's self-training trick (VkNRC's own core idea): resolves each
    /// <see cref="TrainRecord"/>'s regression target as
    /// <c>Target = Bias + Factor * cache(Endpoint)</c> - a forward-only query of the
    /// EMA weights at the training suffix's endpoint, run BEFORE the gradient pass so
    /// the network trains against an (approximately) unbiased estimate of the FULL
    /// remaining path, not just the short suffix's own directly-observed radiance.
    /// Resolves targets for every one of the
    /// <see cref="Core.NrcConstants.TrainBatchesPerFrame"/> disjoint batches in one
    /// pass (matching VkNRC's own <c>nrc_inference.comp</c>, which resolves every
    /// EvalRecord regardless of batch together) - bounds-checks each record against
    /// its OWN batch's live device-resident counter (no host readback.
    /// </summary>
    public struct SelfTrainingTargetLaunchParams
    {
        public OptixDeviceView<TrainRecord> TrainRecords;
        public OptixDeviceView<int> TrainCounter;

        public ulong InferenceWeightsBase;
        public uint HiddenLayerStrideInBytes;
        public uint OutputLayerOffsetInBytes;

        public ulong ZeroBias64Base;
        public ulong ZeroBias3Base;
        public ulong ZeroVec64Base;

        public ulong ActScratchBase;
        public OptixDeviceView<Half> ActScratch;
        public ulong OutputScratchBase;
        public OptixDeviceView<Half> OutputScratch;

        /// <summary>Network depth - see GradientLaunchParams.HiddenLayerCount's own doc comment.</summary>
        public int HiddenLayerCount;
        /// <summary>Network width - see GradientLaunchParams.LayerWidth's own doc comment.</summary>
        public int LayerWidth;
    }

    public static class SelfTrainingTargetProgram
    {
        /// <summary>Generic over the network-width strategy - see GradientProgram.ComputeGradient{TNet}'s own doc comment.</summary>
        public static void ResolveTarget<TNet>(SelfTrainingTargetLaunchParams p) where TNet : struct, INrcNetworkOps
        {
            long recordIdx = OptixGetLaunchIndex.X;
            if (recordIdx >= p.TrainRecords.Length)
                return;

            // Which of the TrainBatchesPerFrame disjoint slices this global index falls
            // in, and the live count for THAT batch specifically (see TrainRecordBuffers'
            // own doc comment for the batch-slicing scheme).
            int batch = (int)(recordIdx / Core.NrcConstants.TrainBatchSize);
            int local = (int)(recordIdx % Core.NrcConstants.TrainBatchSize);
            int count = XMath.Min(p.TrainCounter[batch], Core.NrcConstants.TrainBatchSize);
            if (local >= count)
                return;

            TrainRecord record = p.TrainRecords[recordIdx];

            // Self-training bootstrap: Target = Bias + Factor*cache(Endpoint) (VkNRC's
            // core NRC mechanism). Needed because suffixes truncated early by the
            // adaptive footprint handoff (RaygenProgram.cs's nrcSuffixSqrtASum/nrcC_a0)
            // would otherwise train toward a bias-truncated, systematically too-dark
            // target in an indirect-light-dominated scene (Sponza).
            //
            // Factor == 0 records (suffix ended naturally at BOUNCE_TERMINAL/bounce
            // budget - RaygenProgram emits Factor=(0,0,0) there, nothing left to
            // predict) skip the cache query outright: Target = Bias exactly, and the
            // forward pass's result would be multiplied away anyway.
            if (record.Factor.x != 0f || record.Factor.y != 0f || record.Factor.z != 0f)
            {
                Vec3 dielectricF0 = new Vec3(0.04f, 0.04f, 0.04f);
                Vec3 endpointSpecular =
                    dielectricF0 + ((record.EndpointAlbedo - dielectricF0) * record.EndpointMetallic);

                // slotsPerRecord=2: forward-only ping-pong path, same as
                // CompositeProgram - nothing
                // backpropagates through the bootstrap query, so caching every layer's
                // activation would be pure scratch traffic.
                InputEncoding.Encode(p.ActScratch, recordIdx, record.EndpointPosition,
                    record.EndpointScatteredDir, record.EndpointNormal, record.EndpointRoughness,
                    record.EndpointAlbedo, endpointSpecular, 2, p.LayerWidth);

                default(TNet).ForwardInference(recordIdx, p.ActScratchBase, p.InferenceWeightsBase,
                    p.HiddenLayerStrideInBytes, p.OutputLayerOffsetInBytes, p.ZeroBias64Base,
                    p.ZeroBias3Base, p.ZeroVec64Base, p.OutputScratchBase, p.HiddenLayerCount);

                // Clamped to physical (non-negative) radiance ONLY - the same clamp
                // CompositeProgram/GradientProgram apply (VkNRC's max(pred, 0), no upper
                // bound - see GradientProgram.cs's own note).
                Vec3 cachePrediction = new Vec3(
                    XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 0], 0f),
                    XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 1], 0f),
                    XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 2], 0f));

                record.Target = record.Bias + (record.Factor * cachePrediction);
            }
            else
            {
                record.Target = record.Bias;
            }
            p.TrainRecords[recordIdx] = record;
        }
    }
}
