// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: ForwardKernel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.DeviceApi;
using Sample21.Core;

namespace Sample21.Device.Network
{
    /// <summary>
    /// Forward pass through the 5x(64-&gt;64 ReLU) + (64-&gt;3 linear) MLP, reused by
    /// three real pipeline stages : self-training-target
    /// resolution and composite (both <see cref="Training"/> against a
    /// TrainingOptimal-converted EMA snapshot, forward-only reads), and the gradient
    /// pass's own forward half (<see cref="Training"/> against the live training
    /// weights, immediately followed by <see cref="BackwardKernel"/>). No
    /// InferencingOptimal-layout entry point exists - see
    /// WeightBuffers.ConvertMasterToTrainingAndInference's own note for why
    /// (TrainingOptimal is a valid, just not perf-optimal, layout for a forward-only
    /// read too).
    /// </summary>
    public static class ForwardKernel
    {
        private const uint HalfSize = 2;

        /// <summary>Forward pass against the TrainingOptimal (live) weight buffer - caches every layer's activation in act-scratch for an immediately-following <see cref="BackwardKernel"/> call.</summary>
        public static void Training(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBias64Base,
            ulong zeroBias3Base,
            ulong zeroVec64Base,
            ulong outputScratchBase,
            int hiddenLayerCount)
        {
            ulong actBase = actScratchBase + ((ulong)recordIdx * (ulong)(hiddenLayerCount + 1) * NrcConstants.LayerWidth * HalfSize);
            ulong outputAddr = outputScratchBase + ((ulong)recordIdx * NrcConstants.OutputWidth * HalfSize);

            for (int h = 0; h < hiddenLayerCount; h++)
            {
                ulong inAddr = actBase + ((ulong)h * NrcConstants.LayerWidth * HalfSize);
                ulong outAddr = actBase + ((ulong)(h + 1) * NrcConstants.LayerWidth * HalfSize);
                uint layerOffset = (uint)h * hiddenLayerStrideInBytes;
                OptixCoopVec.MatVecMul_N64_K64_Training(inAddr, weightsBase, layerOffset, zeroBias64Base, 0, outAddr);
                OptixCoopVec.Max_S64(outAddr, zeroVec64Base, outAddr);
            }
            ulong lastHiddenAddr = actBase + ((ulong)hiddenLayerCount * NrcConstants.LayerWidth * HalfSize);
            OptixCoopVec.MatVecMul_N3_K64_Training(lastHiddenAddr, weightsBase, outputLayerOffsetInBytes, zeroBias3Base, 0, outputAddr);
        }

        /// <summary>
        /// Forward-only variant for inference call sites (composite/self-training-target
        /// resolution): ping-pongs between TWO act-scratch slots per record instead of
        /// caching every layer's activation, since nothing backpropagates through these
        /// passes. The per-record act footprint
        /// drops from (hiddenLayerCount+1) slots to 2 - at the composite's
        /// one-record-per-pixel launch that's the difference between a
        /// pixels*(layers+1)*width scratch allocation and pixels*2*width, and a much
        /// hotter working set per thread. The caller's InputEncoding.Encode must have
        /// used slotsPerRecord=2 for the addressing to line up.
        /// </summary>
        public static void Inference(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBias64Base,
            ulong zeroBias3Base,
            ulong zeroVec64Base,
            ulong outputScratchBase,
            int hiddenLayerCount)
        {
            ulong slot0 = actScratchBase + ((ulong)recordIdx * 2 * NrcConstants.LayerWidth * HalfSize);
            ulong slot1 = slot0 + (NrcConstants.LayerWidth * HalfSize);
            ulong outputAddr = outputScratchBase + ((ulong)recordIdx * NrcConstants.OutputWidth * HalfSize);

            ulong inAddr = slot0;
            ulong outAddr = slot1;
            for (int h = 0; h < hiddenLayerCount; h++)
            {
                uint layerOffset = (uint)h * hiddenLayerStrideInBytes;
                OptixCoopVec.MatVecMul_N64_K64_Training(inAddr, weightsBase, layerOffset, zeroBias64Base, 0, outAddr);
                OptixCoopVec.Max_S64(outAddr, zeroVec64Base, outAddr);
                ulong swap = inAddr;
                inAddr = outAddr;
                outAddr = swap;
            }
            OptixCoopVec.MatVecMul_N3_K64_Training(inAddr, weightsBase, outputLayerOffsetInBytes, zeroBias3Base, 0, outputAddr);
        }
    }
}
