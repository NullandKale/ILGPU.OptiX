// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: BackwardKernel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.DeviceApi;
using Sample21.Core;

namespace Sample21.Device.Network
{
    /// <summary>
    /// Backward pass for the same 5x(64-&gt;64 ReLU) + (64-&gt;3 linear) MLP, called
    /// immediately after <see cref="ForwardKernel.Training"/> in the gradient pass (the
    /// two are fused into one "gradient" dispatch, matching VkNRC's own NN_nv.glsl
    /// convention). The real pipeline's gradient kernel supplies a relative-L2-luminance
    /// loss gradient against <see cref="TrainRecord.Target"/>.
    /// </summary>
    public static class BackwardKernel
    {
        private const uint HalfSize = 2;

        /// <summary>
        /// Backpropagates a loss gradient already written to
        /// <paramref name="daOutScratchBase"/>[recordIdx] (3-wide) through the network
        /// whose forward activations <see cref="ForwardKernel.Training"/> just cached in
        /// <paramref name="actScratchBase"/>, accumulating into
        /// <paramref name="dwBase"/> (TrainingOptimal Float16, zeroed once per training
        /// batch by the caller).
        /// </summary>
        public static void Backward(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            ulong dwBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBias64Base,
            ulong zeroVec64Base,
            ulong daOutScratchBase,
            ulong daScratchBase,
            ulong dInScratchBase,
            ulong maskScratchBase,
            int hiddenLayerCount)
        {
            ulong actBase = actScratchBase + ((ulong)recordIdx * (ulong)(hiddenLayerCount + 1) * NrcConstants.LayerWidth * HalfSize);
            ulong daOutAddr = daOutScratchBase + ((ulong)recordIdx * NrcConstants.OutputWidth * HalfSize);
            ulong daAddr = daScratchBase + ((ulong)recordIdx * NrcConstants.LayerWidth * HalfSize);
            ulong dInAddr = dInScratchBase + ((ulong)recordIdx * NrcConstants.LayerWidth * HalfSize);
            ulong maskAddr = maskScratchBase + ((ulong)recordIdx * NrcConstants.LayerWidth * HalfSize);
            ulong lastHiddenAddr = actBase + ((ulong)hiddenLayerCount * NrcConstants.LayerWidth * HalfSize);

            // Output layer. The ReLU derivative needs a STRICT act > 0 test: the cached
            // activations are post-ReLU (ForwardKernel applies Max in place), so dead
            // units are exactly 0 and a >=-0 test (Step's semantics) would pass them -
            // making the mask identically 1 and backprop linear. VkNRC's own
            // _nn_act_64_relu_mask_t (NN_nv.glsl) uses strict `act > 0.0` via a
            // NaN-poison; the op2 set has no strict compare, so build the DEAD mask
            // (1 where 0 >= act, i.e. act == 0) and subtract its components instead.
            OptixCoopVec.OuterProductAccumulate_SA3_SB64(daOutAddr, lastHiddenAddr, dwBase, outputLayerOffsetInBytes);
            OptixCoopVec.MatVecMul_N64_K3_TrainingTranspose(daOutAddr, weightsBase, outputLayerOffsetInBytes, zeroBias64Base, 0, dInAddr);
            OptixCoopVec.Step_S64(lastHiddenAddr, zeroVec64Base, maskAddr); // 1 where act[5] == 0 (dead)
            OptixCoopVec.Mul_S64(dInAddr, maskAddr, maskAddr);              // dead components of dIn
            OptixCoopVec.Sub_S64(dInAddr, maskAddr, daAddr);                // live-only gradient

            // Hidden layers (hiddenLayerCount-1)..0.
            for (int h = hiddenLayerCount - 1; h >= 0; h--)
            {
                ulong inActAddr = actBase + ((ulong)h * NrcConstants.LayerWidth * HalfSize);
                uint layerOffset = (uint)h * hiddenLayerStrideInBytes;
                OptixCoopVec.OuterProductAccumulate_SA64_SB64(daAddr, inActAddr, dwBase, layerOffset);

                if (h > 0)
                {
                    // Same dead-mask-and-subtract as the output layer above.
                    OptixCoopVec.MatVecMul_N64_K64_TrainingTranspose(daAddr, weightsBase, layerOffset, zeroBias64Base, 0, dInAddr);
                    OptixCoopVec.Step_S64(inActAddr, zeroVec64Base, maskAddr);
                    OptixCoopVec.Mul_S64(dInAddr, maskAddr, maskAddr);
                    OptixCoopVec.Sub_S64(dInAddr, maskAddr, daAddr);
                }
            }
        }
    }
}
