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

namespace Sample21.Core
{
    /// <summary>
    /// Every "magic number" for the VkNRC-style MLP: layer widths, Adam/EMA constants,
    /// per-frame training batch sizing. Ported from VkNRC (example/VkNRC/shader/src/
    /// NN_nv.glsl and Constant.glsl).
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

        // Adam (VkNRC Constant.glsl): beta1=0.9, beta2=0.999, eps=1e-8, bias-corrected.
        public const float AdamBeta1 = 0.9f;
        public const float AdamBeta2 = 0.999f;
        // VkNRC's own tuned value. GradientProgram.cs/SelfTrainingTargetProgram.cs/
        // CompositeProgram.cs all clamp the network's output to non-negative before it
        // reaches the loss, so this doesn't need a lower workaround value; the live UI
        // learning-rate slider is available as a fallback if it proves unstable.
        public const float AdamLearningRate = 0.0001f;
        public const float AdamEpsilon = 1e-8f;

        /// <summary>
        /// AdamW (Loshchilov &amp; Hutter 2019) decoupled weight-decay coefficient - not
        /// part of VkNRC's original plain-Adam algorithm; pulls weight magnitude back
        /// down over time. Tune via the live UI slider (SampleRenderer.NrcWeightDecay)
        /// without a rebuild.
        /// </summary>
        public const float AdamWeightDecay = 1e-3f;

        /// <summary>
        /// Rescales each record's raw loss gradient (GradientProgram.cs) BEFORE
        /// BackwardKernel.Backward accumulates it into DwConverted - a Float16
        /// TrainingOptimal buffer (OptixCoopVec.OuterProductAccumulate can only write
        /// that layout/element-type, a hard SDK constraint), whose ~65504 max
        /// representable value would otherwise overflow when up to
        /// TrainBatchSize=16,384 records' unscaled gradients sum into one FP16
        /// accumulator. AdamKernel.AdamStep divides the accumulated sum back out by this
        /// same constant (in addition to the live-batchCount divisor) to recover the
        /// true mean gradient.
        /// </summary>
        public const float GradientAccumScale = 1f / TrainBatchSize;

        /// <summary>
        /// Max allowed L2 norm of a single record's raw <c>dLoss</c> seed
        /// (GradientProgram.cs), applied BEFORE GradientAccumScale/BackwardKernel.
        /// </summary>
        /// <remarks>
        /// Not a VkNRC mechanism (VkNRC trains on raw HDR radiance with no gradient
        /// clamp of any kind) - a safety net against the relative-L2-luminance loss's
        /// own denominator-collapse regime (undertrained predictions near zero produce
        /// enormous gradients against bright firefly targets). Set high enough (100) to
        /// only engage in that pathological regime and leave ordinary gradient
        /// magnitudes untouched - a lower bound turns the optimizer into median
        /// regression, since path-traced radiance per region is heavily right-skewed
        /// (median far below mean).
        /// </remarks>
        public const float MaxGradientNorm = 100f;

        /// <summary>
        /// EMA weight snapshot decay. Computed every Adam step (AdamKernel.cs) but NOT
        /// what the renderer samples from by default - WeightBuffers.
        /// ConvertMasterToTrainingAndInference reads MasterWeights (live) for
        /// composite/self-training-target inference, matching VkNRC's own default
        /// (m_use_ema_weights{false} in VkNRCState.hpp). SampleRenderer.NrcUseEmaInference
        /// (UiPanel's "Use EMA weights" checkbox) switches to the EMA snapshot; this
        /// value only affects rendered output while that toggle is on.
        /// </summary>
        // VkNRC's own value, exposed live in the UI (UI/UiPanel.cs's NRC section,
        // SampleRenderer.NrcEmaAlpha) instead of hardcoded, so it can be swept without
        // a rebuild.
        public const float EmaAlpha = 0.99f;

        /// <summary>Per-frame training batch size (VkNRC: 4 batches of 16,384 records/frame).</summary>
        public const int TrainBatchSize = 16384;

        /// <summary>Number of training batches (gradient+optimize iterations) per frame.</summary>
        public const int TrainBatchesPerFrame = 4;

        /// <summary>Max train records buffered per frame - one pool sized for the worst case, not reallocated.</summary>
        public const int MaxTrainRecords = TrainBatchSize * TrainBatchesPerFrame;

        /// <summary>
        /// VkNRC's adaptive ray-footprint handoff constant ("C" in path_tracer.comp:
        /// <c>c_a0 = C * dist2 / (4*PI*cosine)</c>). Replaces the earlier fixed
        /// NrcHandoffBounce constant - the handoff bounce is now decided per-path by
        /// comparing accumulated footprint growth against this threshold
        /// (RaygenProgram.cs), matching VkNRC's own design exactly rather than a fixed
        /// bounce index.
        /// </summary>
        public const float FootprintConstant = 0.01f;

        /// <summary>
        /// Probability, per eval-record pixel, of also tracing a short training suffix
        /// (VkNRC: ~3% of pixels/frame) to generate a self-training target.
        /// </summary>
        public const float TrainProbability = 0.03f;

        /// <summary>Per-layer element offset (into a flat RowMajor FP32 weight/dW/moment array) for hidden layer <paramref name="layerIndex"/> (0-4) or the output layer (index 5).</summary>
        public static int RowMajorLayerOffset(int layerIndex) => RowMajorLayerOffset(layerIndex, HiddenLayerCount);

        /// <summary>
        /// Same as <see cref="RowMajorLayerOffset(int)"/> but for an arbitrary hidden-layer
        /// count at the default width (64) - used by <see cref="Rendering.WeightBuffers"/> when
        /// constructed for a swept depth.
        /// </summary>
        public static int RowMajorLayerOffset(int layerIndex, int hiddenLayerCount) => RowMajorLayerOffset(layerIndex, hiddenLayerCount, (int)LayerWidth);

        /// <summary>
        /// Same as <see cref="RowMajorLayerOffset(int, int)"/> but for an arbitrary width
        /// too - used by <see cref="Rendering.WeightBuffers"/> when constructed for a
        /// swept width.
        /// </summary>
        public static int RowMajorLayerOffset(int layerIndex, int hiddenLayerCount, int layerWidth)
        {
            if (layerIndex < 0 || layerIndex > hiddenLayerCount)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            return layerIndex < hiddenLayerCount
                ? layerIndex * layerWidth * layerWidth
                : hiddenLayerCount * layerWidth * layerWidth;
        }

        /// <summary>Total flat RowMajor FP32 weight element count for an arbitrary hidden-layer count at the default width (64) - see <see cref="TotalWeightCount"/>'s own doc comment.</summary>
        public static int ComputeTotalWeightCount(int hiddenLayerCount) => ComputeTotalWeightCount(hiddenLayerCount, (int)LayerWidth);

        /// <summary>
        /// Same as <see cref="ComputeTotalWeightCount(int)"/> but for an arbitrary width
        /// too - used by <see cref="Rendering.WeightBuffers"/> when constructed for a
        /// swept width.
        /// </summary>
        public static int ComputeTotalWeightCount(int hiddenLayerCount, int layerWidth) =>
            (hiddenLayerCount * layerWidth * layerWidth) + ((int)OutputWidth * layerWidth);
    }
}
