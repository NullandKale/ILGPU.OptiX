// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: NrcConstants.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

namespace Sample25.Core
{
    /// <summary>
    /// Every "magic number" for the VkNRC-style MLP: layer widths, Adam/EMA constants,
    /// per-frame training batch sizing. Ported from VkNRC (example/VkNRC/shader/src/
    /// NN_nv.glsl and Constant.glsl) - see docs/SAMPLE25_PLAN.md for the port rationale.
    /// </summary>
    public static class NrcConstants
    {
        /// <summary>Width of every hidden layer and the network's input vector.</summary>
        public const uint LayerWidth = 64;

        /// <summary>Number of 64-&gt;64 ReLU hidden layers (the output layer is separate).</summary>
        public const int HiddenLayerCount = 5;

        /// <summary>Width of the (linear, no activation) output layer.</summary>
        public const uint OutputWidth = 3;

        /// <summary>Total layer count including the output layer (5 hidden + 1 output).</summary>
        public const int LayerCount = HiddenLayerCount + 1;

        /// <summary>
        /// Total FP16 weight element count across every layer - 5*64*64 (hidden) +
        /// 3*64 (output) = 20,672, matching VkNRC's network size exactly (no biases).
        /// </summary>
        public const int TotalWeightCount = HiddenLayerCount * (int)LayerWidth * (int)LayerWidth + (int)OutputWidth * (int)LayerWidth;

        // Adam (VkNRC Constant.glsl): beta1=0.9, beta2=0.999, lr=0.002, eps=1e-8, bias-corrected.
        public const float AdamBeta1 = 0.9f;
        public const float AdamBeta2 = 0.999f;
        public const float AdamLearningRate = 0.002f;
        public const float AdamEpsilon = 1e-8f;

        /// <summary>EMA weight snapshot decay - the renderer samples from this copy, not the live training weights.</summary>
        public const float EmaAlpha = 0.99f;

        /// <summary>Per-frame training batch size (VkNRC: 4 batches of 16,384 records/frame).</summary>
        public const int TrainBatchSize = 16384;

        /// <summary>Number of training batches (gradient+optimize iterations) per frame.</summary>
        public const int TrainBatchesPerFrame = 4;

        /// <summary>Max train records buffered per frame - one pool sized for the worst case, not reallocated.</summary>
        public const int MaxTrainRecords = TrainBatchSize * TrainBatchesPerFrame;

        /// <summary>Per-layer element offset (into a flat RowMajor FP32 weight/dW/moment array) for hidden layer <paramref name="layerIndex"/> (0-4) or the output layer (index 5).</summary>
        public static int RowMajorLayerOffset(int layerIndex)
        {
            if (layerIndex < 0 || layerIndex >= LayerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            return layerIndex < HiddenLayerCount
                ? layerIndex * (int)LayerWidth * (int)LayerWidth
                : HiddenLayerCount * (int)LayerWidth * (int)LayerWidth;
        }
    }
}
