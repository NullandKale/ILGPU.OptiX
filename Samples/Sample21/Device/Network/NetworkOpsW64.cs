// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: NetworkOpsW64.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample21.Device.Network
{
    /// <summary>
    /// <see cref="INrcNetworkOps"/> for the default 64-wide network - a thin wrapper
    /// around <see cref="ForwardKernel.Training"/>/<see cref="BackwardKernel.Backward"/>.
    /// This is the network-width sweep's control/baseline case.
    /// </summary>
    public readonly struct NetworkOpsW64 : INrcNetworkOps
    {
        public void ForwardInference(
            long recordIdx, ulong actScratchBase, ulong weightsBase, uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes, ulong zeroBiasHiddenBase, ulong zeroBias3Base, ulong zeroVecHiddenBase,
            ulong outputScratchBase, int hiddenLayerCount) =>
            ForwardKernel.Inference(recordIdx, actScratchBase, weightsBase, hiddenLayerStrideInBytes,
                outputLayerOffsetInBytes, zeroBiasHiddenBase, zeroBias3Base, zeroVecHiddenBase,
                outputScratchBase, hiddenLayerCount);

        public void Forward(
            long recordIdx, ulong actScratchBase, ulong weightsBase, uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes, ulong zeroBiasHiddenBase, ulong zeroBias3Base, ulong zeroVecHiddenBase,
            ulong outputScratchBase, int hiddenLayerCount) =>
            ForwardKernel.Training(recordIdx, actScratchBase, weightsBase, hiddenLayerStrideInBytes,
                outputLayerOffsetInBytes, zeroBiasHiddenBase, zeroBias3Base, zeroVecHiddenBase,
                outputScratchBase, hiddenLayerCount);

        public void Backward(
            long recordIdx, ulong actScratchBase, ulong weightsBase, ulong dwBase, uint hiddenLayerStrideInBytes,
            uint outputLayerOffsetInBytes, ulong zeroBiasHiddenBase, ulong zeroVecHiddenBase,
            ulong daOutScratchBase, ulong daScratchBase, ulong dInScratchBase, ulong maskScratchBase, int hiddenLayerCount) =>
            BackwardKernel.Backward(recordIdx, actScratchBase, weightsBase, dwBase, hiddenLayerStrideInBytes,
                outputLayerOffsetInBytes, zeroBiasHiddenBase, zeroVecHiddenBase,
                daOutScratchBase, daScratchBase, dInScratchBase, maskScratchBase, hiddenLayerCount);
    }
}
