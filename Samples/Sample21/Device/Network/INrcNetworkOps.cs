// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: INrcNetworkOps.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample21.Device.Network
{
    /// <summary>
    /// Forward/backward pass for one network WIDTH - unlike <see cref="AdamKernel"/>'s optimizer choice or
    /// <see cref="ForwardKernel"/>/<see cref="BackwardKernel"/>'s <c>hiddenLayerCount</c>
    /// (an ordinary runtime loop bound), the layer WIDTH is baked into named
    /// <c>OptixCoopVec.MatVecMul_N{width}_K{width}...</c>/<c>OuterProductAccumulate_SA{width}_SB{width}</c>
    /// calls - a genuine PTX compile-time constant OptiX's cooperative-vector API
    /// requires (see OptixCoopVec.tt's own class doc comment). <see cref="GradientProgram"/>/
    /// <see cref="SelfTrainingTargetProgram"/>/<see cref="CompositeProgram"/> are generic
    /// over <c>TNet : struct, INrcNetworkOps</c> so each width's implementation (a small
    /// struct closing over its own named CoopVec calls - see <see cref="NetworkOpsW64"/>/
    /// <see cref="NetworkOpsW32"/>/<see cref="NetworkOpsW128"/>) gets compiled into its own
    /// concrete OptiX pipeline (<see cref="Rendering.NrcPipelines"/> picks the closed
    /// generic method per width) without duplicating the three device Program classes
    /// once per width.
    /// </summary>
    public interface INrcNetworkOps
    {
        /// <summary>
        /// Forward-only pass for inference call sites (composite/self-training-target)
        /// - ping-pongs 2 act-scratch slots per record instead of caching every layer
        /// like <see cref="Forward"/> does for backprop; see
        /// <see cref="ForwardKernel.Inference"/>'s own doc comment. Callers must encode
        /// the input with <c>slotsPerRecord=2</c>.
        /// </summary>
        void ForwardInference(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBiasHiddenBase,
            ulong zeroBias3Base,
            ulong zeroVecHiddenBase,
            ulong outputScratchBase,
            int hiddenLayerCount);

        /// <summary>Same parameter order/meaning as <see cref="ForwardKernel.Training"/>.</summary>
        void Forward(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBiasHiddenBase,
            ulong zeroBias3Base,
            ulong zeroVecHiddenBase,
            ulong outputScratchBase,
            int hiddenLayerCount);

        /// <summary>Same parameter order/meaning as <see cref="BackwardKernel.Backward"/>.</summary>
        void Backward(
            long recordIdx,
            ulong actScratchBase,
            ulong weightsBase,
            ulong dwBase,
            uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes,
            ulong zeroBiasHiddenBase,
            ulong zeroVecHiddenBase,
            ulong daOutScratchBase,
            ulong daScratchBase,
            ulong dInScratchBase,
            ulong maskScratchBase,
            int hiddenLayerCount);
    }
}
