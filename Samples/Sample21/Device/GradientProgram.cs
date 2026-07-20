// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: GradientProgram.cs
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
    /// The "gradient" dispatch: fused forward (<see cref="ForwardKernel.Training"/>,
    /// live TrainingOptimal weights) + relative-L2-luminance loss (VkNRC's
    /// NNLoadDA3_RelativeL2LuminanceLoss) against <see cref="TrainRecord.Target"/>
    /// (already resolved by <see cref="SelfTrainingTargetProgram"/>) +
    /// <see cref="BackwardKernel.Backward"/>, accumulating into
    /// <see cref="Rendering.WeightBuffers.DwConverted"/>. Launched
    /// <see cref="Core.NrcConstants.TrainBatchesPerFrame"/> times per frame, each call
    /// against ONE of the 4 disjoint <see cref="BatchIndex"/> record slices (VkNRC's
    /// own 4-different-batches-per-frame scheme, chained weight updates - see
    /// TrainRecordBuffers's own doc comment for the slicing), not 4 epochs over the
    /// same live pool.
    /// </summary>
    public struct GradientLaunchParams
    {
        public OptixDeviceView<TrainRecord> TrainRecords;
        public OptixDeviceView<int> TrainCounter;
        // Which of the TrainBatchesPerFrame disjoint slices this launch processes -
        // grid size is TrainBatchSize threads (not the full pool), each thread's
        // global record index is BatchIndex*TrainBatchSize + localIdx.
        public int BatchIndex;

        public ulong TrainingWeightsBase;
        public ulong DwBase;
        public uint HiddenLayerStrideInBytes;
        public uint OutputLayerOffsetInBytes;
        /// <summary>Network depth - a runtime loop bound for ForwardKernel.Training/BackwardKernel.Backward, not baked into any OptixCoopVec call.</summary>
        public int HiddenLayerCount;
        /// <summary>Network width - passed to InputEncoding.Encode (a plain runtime parameter there); the CoopVec-call width itself is fixed by which <c>TNet</c> ComputeGradient&lt;TNet&gt; was compiled for, see NrcPipelines.cs's own doc comment.</summary>
        public int LayerWidth;

        public ulong ZeroBias64Base;
        public ulong ZeroBias3Base;
        public ulong ZeroVec64Base;

        public ulong ActScratchBase;
        public OptixDeviceView<Half> ActScratch;
        public ulong OutputScratchBase;
        public OptixDeviceView<Half> OutputScratch;
        public ulong DaOutScratchBase;
        public OptixDeviceView<Half> DaOutScratch;
        public ulong DaScratchBase;
        public ulong DInScratchBase;
        public ulong MaskScratchBase;
    }

    public static class GradientProgram
    {
        /// <summary>
        /// Generic over the network-width strategy <typeparamref name="TNet"/> -
        /// <see cref="Rendering.NrcPipelines"/> compiles one closed-generic
        /// raygen kernel per swept width (<c>ComputeGradient&lt;NetworkOpsW32&gt;</c>,
        /// <c>&lt;NetworkOpsW64&gt;</c>, <c>&lt;NetworkOpsW128&gt;</c>) rather than
        /// duplicating this whole method body per width.
        /// </summary>
        public static void ComputeGradient<TNet>(GradientLaunchParams p) where TNet : struct, INrcNetworkOps
        {
            long localIdx = OptixGetLaunchIndex.X;
            if (localIdx >= Core.NrcConstants.TrainBatchSize)
                return;

            int count = XMath.Min(p.TrainCounter[p.BatchIndex], Core.NrcConstants.TrainBatchSize);
            if (localIdx >= count)
                return;

            long recordIdx = ((long)p.BatchIndex * Core.NrcConstants.TrainBatchSize) + localIdx;
            if (recordIdx >= p.TrainRecords.Length)
                return;

            TrainRecord record = p.TrainRecords[recordIdx];

            Vec3 dielectricF0 = new Vec3(0.04f, 0.04f, 0.04f);
            Vec3 startSpecular = dielectricF0 + ((record.StartAlbedo - dielectricF0) * record.StartMetallic);

            // slotsPerRecord = HiddenLayerCount+1: the TRAINING path keeps the full
            // per-layer activation cache - BackwardKernel needs every layer's forward
            // activation, unlike the composite/self-training inference call sites'
            // 2-slot ping-pong.
            InputEncoding.Encode(p.ActScratch, recordIdx, record.StartPosition, record.StartScatteredDir,
                record.StartNormal, record.StartRoughness, record.StartAlbedo, startSpecular, p.HiddenLayerCount + 1, p.LayerWidth);

            default(TNet).Forward(recordIdx, p.ActScratchBase, p.TrainingWeightsBase, p.HiddenLayerStrideInBytes,
                p.OutputLayerOffsetInBytes, p.ZeroBias64Base, p.ZeroBias3Base, p.ZeroVec64Base, p.OutputScratchBase, p.HiddenLayerCount);

            // Clamped to physical (non-negative) radiance ONLY - matches VkNRC's own
            // `predict = max(NNOutput3(...), vec3(0))` (nrc_inference.comp) exactly, no
            // upper bound. A channel left free to drift negative would contribute 0 to
            // predictLuminance (which only sums max(x,0) per channel) regardless of how
            // negative it got, so its gradient would escape the loss's own damping
            // entirely and diverge unbounded.
            Vec3 predict = new Vec3(
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 0], 0f),
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 1], 0f),
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 2], 0f));

            // VkNRC's NNLoadDA3_RelativeL2LuminanceLoss: weights the L2 gradient down
            // for already-bright predictions, up for near-zero ones - keeps a single
            // learning rate reasonable across the huge dynamic range of path-traced
            // radiance. loss_scale = VkNRC's LOSS_SCALE = 1.0 (Constant.glsl) - VkNRC
            // does NOT divide by batch size here; that division happens later, by the
            // ACTUAL live record count, in AdamKernel.AdamStep (matching
            // nrc_optimize.comp:36's `gradient = uGradients[i] / uBatchTrainCount /
            // LOSS_SCALE`) - not a fixed capacity divisor applied per-record here.
            float predictLuminance = (0.299f * XMath.Max(predict.x, 0f)) + (0.587f * XMath.Max(predict.y, 0f)) + (0.114f * XMath.Max(predict.z, 0f));
            float denom = (predictLuminance * predictLuminance) + 0.01f;
            Vec3 dLoss = (2f / denom) * (predict - record.Target);

            // NOT a VkNRC mechanism - see Core.NrcConstants.MaxGradientNorm's own note
            // for why this is here despite the reference having no equivalent clamp.
            // Bounds the raw loss-gradient seed's L2 norm before it propagates through
            // BackwardKernel.Backward, guarding against HDR fireflies in
            // record.Target (this scene's emissive lights/environment) blowing the
            // relative-L2-luminance loss's own damping past what an undertrained
            // (predict~0) network can absorb.
            float dLossNormSq = Vec3.dot(dLoss, dLoss);
            float maxNormSq = Core.NrcConstants.MaxGradientNorm * Core.NrcConstants.MaxGradientNorm;
            if (dLossNormSq > maxNormSq)
                dLoss *= XMath.Sqrt(maxNormSq / dLossNormSq);

            // Core.NrcConstants.GradientAccumScale's own note - BackwardKernel.Backward
            // accumulates this seed through the whole network into a Float16
            // TrainingOptimal buffer via OuterProductAccumulate, which overflows to
            // +-Infinity if up to TrainBatchSize records' unscaled contributions sum
            // into it. Pre-scaling the seed here (linear through every downstream
            // OuterProductAccumulate/MatVecMul - only Step_S64's ReLU mask is
            // scale-invariant, since it reads the cached forward activation, not this
            // propagated gradient) keeps the accumulator's magnitude bounded regardless
            // of live batch occupancy. AdamKernel.AdamStep divides it back out.
            Vec3 scaledDLoss = dLoss * Core.NrcConstants.GradientAccumScale;
            p.DaOutScratch[(recordIdx * 3) + 0] = (Half)scaledDLoss.x;
            p.DaOutScratch[(recordIdx * 3) + 1] = (Half)scaledDLoss.y;
            p.DaOutScratch[(recordIdx * 3) + 2] = (Half)scaledDLoss.z;

            default(TNet).Backward(recordIdx, p.ActScratchBase, p.TrainingWeightsBase, p.DwBase,
                p.HiddenLayerStrideInBytes, p.OutputLayerOffsetInBytes, p.ZeroBias64Base, p.ZeroVec64Base,
                p.DaOutScratchBase, p.DaScratchBase, p.DInScratchBase, p.MaskScratchBase, p.HiddenLayerCount);
        }
    }
}
