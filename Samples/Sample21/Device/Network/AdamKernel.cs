// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: AdamKernel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Algorithms;
using Sample21.Core;

namespace Sample21.Device.Network
{
    /// <summary>
    /// Plain ILGPU kernel (no CoopVec calls, no OptiX pipeline) - one thread per weight
    /// element. Reads the RowMajor Float32 gradient mirror
    /// (<see cref="Rendering.WeightBuffers.DwRowMajor"/>), applies a bias-corrected Adam
    /// step to the canonical master weights, and updates the EMA snapshot the renderer
    /// actually samples from. Constants ported from VkNRC's Constant.glsl - see
    /// <see cref="NrcConstants"/>.
    /// </summary>
    public static class AdamKernel
    {
        public static void AdamStep(
            Index1D i,
            ArrayView<float> dw,
            ArrayView<float> m,
            ArrayView<float> v,
            ArrayView<float> masterWeights,
            ArrayView<float> emaWeights,
            ArrayView<int> trainCounter,
            int batchIndex,
            float learningRate,
            float weightDecay,
            // Bias-correction/EMA coefficients depend only on stepCount/emaAlpha, not on
            // the per-thread weight index i - every thread in this launch would
            // otherwise recompute the exact same 4 XMath.Pow calls (a measurable cost at
            // the full weight-vector thread count). SampleRenderer.cs computes these
            // once on the host per optimizer step and broadcasts them in as ordinary
            // kernel arguments instead. See RmsPropKernel/SgdMomentumKernel for the
            // shared emaNewCoeff/emaOldCoeff derivation (VkNRC nrc_optimize.comp's
            // bias-corrected EMA: eta_t = 1-alpha^T, eta_t_1 = 1-alpha^(T-1),
            // ema_t = (1-alpha)/eta_t*w_t + alpha*eta_t_1*ema_(t-1)).
            float adamBiasCorrection1,
            float adamBiasCorrection2,
            float emaNewCoeff,
            float emaOldCoeff)
        {
            // VkNRC's actual normalization (nrc_optimize.comp:36):
            // `gradient = uGradients[i] / float(uBatchTrainCount) / LOSS_SCALE`, i.e. the
            // batch's SUMMED gradient (accumulated at loss_scale=1.0 during backprop -
            // see GradientProgram.cs) divided by its ACTUAL live record count here at the
            // optimizer stage, read directly off the GPU-resident counter (no host
            // readback needed, same as VkNRC's own uBatchTrainCount uniform) - NOT a
            // fixed TrainBatchSize divisor, which would under-scale the gradient by
            // roughly TrainBatchSize/actualCount whenever a batch runs under capacity
            // (the common case - TrainProbability keeps most batches well under 16,384).
            int batchCount = XMath.Max(1, XMath.Min(trainCounter[batchIndex], NrcConstants.TrainBatchSize));
            // dw[i] is the SUM of GradientProgram's pre-scaled (by
            // NrcConstants.GradientAccumScale) per-record contributions - divide that
            // back out here (in addition to the batchCount mean) to recover the true
            // mean gradient. See GradientAccumScale's own doc comment for why the
            // pre-scale exists at all (FP16 accumulator overflow, not a correctness
            // choice).
            float g = dw[i] / batchCount / NrcConstants.GradientAccumScale;
            float newM = NrcConstants.AdamBeta1 * m[i] + (1f - NrcConstants.AdamBeta1) * g;
            float newV = NrcConstants.AdamBeta2 * v[i] + (1f - NrcConstants.AdamBeta2) * g * g;
            m[i] = newM;
            v[i] = newV;

            float mHat = newM / adamBiasCorrection1;
            float vHat = newV / adamBiasCorrection2;

            // Full AdamW (Loshchilov & Hutter 2019, decoupled weight decay) - decay is
            // applied directly to the weight, scaled by learningRate, entirely OUTSIDE
            // the m/v moment estimates (NOT folded into g before the moment update,
            // which would be plain L2 regularization instead and would get warped by
            // Adam's per-parameter adaptive scaling). Nothing else in this loop pulls
            // weight magnitude back down, and unbounded weight growth over a long
            // training session is a structural bias (not accumulation noise), which
            // decoupled decay directly counteracts.
            float w = masterWeights[i] - (learningRate * ((mHat / (XMath.Sqrt(vHat) + NrcConstants.AdamEpsilon)) + (weightDecay * masterWeights[i])));
            masterWeights[i] = w;

            emaWeights[i] = (emaNewCoeff * w) + (emaOldCoeff * emaWeights[i]);
        }
    }
}
