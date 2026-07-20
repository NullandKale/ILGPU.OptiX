// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: WeightBuffers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.OptiX.CooperativeVectors;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Sample21.Core;
using System;
using Half = ILGPU.Half;

namespace Sample21.Rendering
{
    /// <summary>
    /// Owns every weight/gradient/optimizer buffer for the NRC MLP and the
    /// <see cref="OptixCoopVecMatrixBuilder.ConvertMatrices"/> calls that move between
    /// them. Canonical state (master weights, EMA snapshot, Adam moments, the row-major
    /// mirror of the gradient) lives as flat RowMajor Float32 - elementwise-addressable
    /// by the plain ILGPU <see cref="Network.AdamKernel"/>, which has no CoopVec calls
    /// and therefore no OptiX pipeline dependency at all.
    ///
    /// <para>
    /// <see cref="OptixCoopVec.OuterProductAccumulate_SA64_SB64"/>/<c>_SA3_SB64</c> can
    /// only accumulate into a Float16 TrainingOptimal-layout matrix (confirmed from the
    /// generated op's operand list - no separate output-element-type slot exists), so
    /// gradient accumulation itself happens in a Float16 TrainingOptimal scratch buffer
    /// (<see cref="DwConverted"/>) each training batch, then gets converted back to the
    /// canonical RowMajor Float32 mirror (<see cref="DwRowMajor"/>) for Adam to consume.
    /// See "gradient-accumulation precision" note for the full
    /// reasoning - this is a deliberate deviation from VkNRC's own all-FP32-atomic
    /// scheme, forced by OptiX's opaque TrainingOptimal layout having no manually
    /// addressable element offsets.
    /// </para>
    /// </summary>
    public sealed class WeightBuffers : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense> MasterWeights { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> EmaWeights { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> AdamM { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> AdamV { get; }
        public MemoryBuffer1D<float, Stride1D.Dense> DwRowMajor { get; }

        /// <summary>TrainingOptimal Float16 - converted from <see cref="MasterWeights"/> once per training batch.</summary>
        public MemoryBuffer1D<byte, Stride1D.Dense> TrainingWeights { get; }

        /// <summary>TrainingOptimal Float16 gradient accumulator - zeroed then written by <c>OuterProductAccumulate</c> each batch.</summary>
        public MemoryBuffer1D<byte, Stride1D.Dense> DwConverted { get; }

        /// <summary>
        /// TrainingOptimal Float16 (NOT InferencingOptimal - see the constructor's own
        /// note) - converted from <see cref="EmaWeights"/> for pure-forward use
        /// (composite/self-training-target).
        /// </summary>
        public MemoryBuffer1D<byte, Stride1D.Dense> InferenceWeights { get; }

        /// <summary>Byte size of one converted 64x64 TrainingOptimal layer - every hidden layer is this size, so layer h's offset is h * this value.</summary>
        public uint HiddenLayerStrideInBytes { get; }

        /// <summary>Byte offset of the output (3x64) layer within the TrainingOptimal buffer - always <c>HiddenLayerCount * HiddenLayerStrideInBytes</c>.</summary>
        public uint OutputLayerOffsetInBytes { get; }

        /// <summary>Network depth this instance was built for - a runtime construction parameter, not the NrcConstants.HiddenLayerCount default, once swept.</summary>
        public int HiddenLayerCount { get; }

        /// <summary>Network width this instance was built for - a runtime construction parameter, not the NrcConstants.LayerWidth default, once swept. Must match whichever INrcNetworkOps the active NrcPipelines was built for.</summary>
        public int LayerWidth { get; }

        /// <summary>Total flat RowMajor FP32 weight element count for <see cref="HiddenLayerCount"/>/<see cref="LayerWidth"/> - see <see cref="NrcConstants.ComputeTotalWeightCount(int, int)"/>.</summary>
        public int TotalWeightCount { get; }

        // Pre-marshaled conversion plans, built once per instance - the raw
        // deviceContext.ConvertMatrices(...) extension these replace re-marshals both
        // network descriptions (four AllocHGlobal/Free pairs) on every call, and the
        // training loop converts up to several times per trained frame. One plan per
        // direction is enough: a plan describes layouts, not storage, so
        // masterToTrainingOptimal serves the TrainingWeights, InferenceWeights, and
        // EMA conversions alike.
        private readonly OptixCoopVecConversionPlan masterToTrainingOptimal;
        private readonly OptixCoopVecConversionPlan trainingOptimalToRowMajor;

        public WeightBuffers(OptixDeviceContext deviceContext, CudaAccelerator accelerator, Random rng,
            int hiddenLayerCount = NrcConstants.HiddenLayerCount, int layerWidth = (int)NrcConstants.LayerWidth)
        {
            HiddenLayerCount = hiddenLayerCount;
            LayerWidth = layerWidth;
            TotalWeightCount = NrcConstants.ComputeTotalWeightCount(hiddenLayerCount, layerWidth);

            var shapes = new (uint N, uint K)[hiddenLayerCount + 1];
            for (int h = 0; h < hiddenLayerCount; h++)
                shapes[h] = ((uint)layerWidth, (uint)layerWidth);
            shapes[hiddenLayerCount] = (NrcConstants.OutputWidth, (uint)layerWidth);

            var rowMajorFp32Descriptions = BuildDescriptions(deviceContext, shapes, OptixCoopVecElemType.Float32, OptixCoopVecMatrixLayout.RowMajor, out _);
            var trainingOptimalFp16Descriptions = BuildDescriptions(deviceContext, shapes, OptixCoopVecElemType.Float16, OptixCoopVecMatrixLayout.TrainingOptimal, out uint trainingOptimalTotalBytes);
            masterToTrainingOptimal = new OptixCoopVecConversionPlan(
                deviceContext, rowMajorFp32Descriptions, trainingOptimalFp16Descriptions);
            trainingOptimalToRowMajor = new OptixCoopVecConversionPlan(
                deviceContext, trainingOptimalFp16Descriptions, rowMajorFp32Descriptions);

            HiddenLayerStrideInBytes = trainingOptimalFp16Descriptions[0].SizeInBytes;
            OutputLayerOffsetInBytes = trainingOptimalFp16Descriptions[hiddenLayerCount].OffsetInBytes;

            float[] initial = BuildInitialWeights(rng);

            MasterWeights = accelerator.Allocate1D(initial);
            EmaWeights = accelerator.Allocate1D(initial);
            // Allocate1D<T>(count) (no host data) does NOT zero device memory (this
            // codebase's own established pattern - see ZeroDw()/FrameOutput.cs's own
            // MemSetToZero calls) - left uninitialized, Adam's first step reads garbage
            // moment estimates here, which can be a NaN bit pattern that poisons every
            // weight it touches permanently (m/v feed w every step thereafter).
            AdamM = accelerator.Allocate1D<float>(TotalWeightCount);
            AdamM.MemSetToZero();
            AdamV = accelerator.Allocate1D<float>(TotalWeightCount);
            AdamV.MemSetToZero();
            DwRowMajor = accelerator.Allocate1D<float>(TotalWeightCount);

            TrainingWeights = accelerator.Allocate1D<byte>((int)trainingOptimalTotalBytes);
            DwConverted = accelerator.Allocate1D<byte>((int)trainingOptimalTotalBytes);
            // Sized/laid out identically to TrainingWeights (both TrainingOptimal) - see
            // ConvertMasterToTrainingAndInference's own note on why this is
            // TrainingOptimal, not the (removed) InferencingOptimal path.
            InferenceWeights = accelerator.Allocate1D<byte>((int)trainingOptimalTotalBytes);
        }

        // Kaiming/He-uniform for the ReLU hidden layers: bound = sqrt(6/fan_in) so a
        // UNIFORM draw in [-bound,bound] has variance 2/fan_in (matching the Gaussian
        // He-init target). A uniform draw scaled by sqrt(2/fan_in) directly has
        // variance (2/fan_in)/3, underscaled by 3x - undersized variance here causes
        // activations to collapse to exactly zero after several stacked ReLU layers
        // regardless of input, a dead network at init that gradient descent cannot
        // recover from (a dead ReLU unit's gradient is also zero). Small fixed range
        // for the linear output layer (no activation to protect against).
        private float[] BuildInitialWeights(Random rng)
        {
            var initial = new float[TotalWeightCount];
            for (int h = 0; h <= HiddenLayerCount; h++)
            {
                int offset = NrcConstants.RowMajorLayerOffset(h, HiddenLayerCount, LayerWidth);
                int count = h < HiddenLayerCount
                    ? LayerWidth * LayerWidth
                    : (int)NrcConstants.OutputWidth * LayerWidth;
                float scale = h < HiddenLayerCount ? MathF.Sqrt(6f / LayerWidth) : 0.1f;
                for (int i = 0; i < count; i++)
                    initial[offset + i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
            }
            return initial;
        }

        /// <summary>
        /// Re-randomizes Master/Ema weights and zeroes the Adam moment buffers - called
        /// on every scene switch (SampleRenderer.SwitchToScene). The cache is a function
        /// of ONE scene's geometry/lighting; carrying a trained network into a different
        /// scene means starting from a wrong radiance field with stale second-moment
        /// estimates suppressing the learning rate exactly where the old scene had large
        /// gradients. The caller must also reset its own Adam step counter so the
        /// Adam/EMA bias corrections restart with the fresh moments.
        /// </summary>
        public void ResetTrainingState(Random rng)
        {
            float[] initial = BuildInitialWeights(rng);
            MasterWeights.CopyFromCPU(initial);
            EmaWeights.CopyFromCPU(initial);
            AdamM.MemSetToZero();
            AdamV.MemSetToZero();
        }

        private static OptixCoopVecMatrixDescription[] BuildDescriptions(
            OptixDeviceContext deviceContext, (uint N, uint K)[] shapes, OptixCoopVecElemType elemType, OptixCoopVecMatrixLayout layout, out uint totalBytes)
        {
            var descriptions = new OptixCoopVecMatrixDescription[shapes.Length];
            uint offset = 0;
            for (int i = 0; i < shapes.Length; i++)
            {
                uint size = deviceContext.ComputeMatrixSizeInBytes(shapes[i].N, shapes[i].K, elemType, layout);
                descriptions[i] = new OptixCoopVecMatrixDescription
                {
                    N = shapes[i].N,
                    K = shapes[i].K,
                    OffsetInBytes = offset,
                    ElementType = elemType,
                    Layout = layout,
                    RowColumnStrideInBytes = 0,
                    SizeInBytes = size,
                };
                offset += size;
            }
            totalBytes = offset;
            return descriptions;
        }

        internal void ConvertMasterToTrainingAndInference(AcceleratorStream stream, bool useEmaForInference = false)
        {
            masterToTrainingOptimal.Convert(
                stream, numNetworks: 1,
                MasterWeights, inputNetworkStrideInBytes: 0,
                TrainingWeights, outputNetworkStrideInBytes: 0);
            ConvertMasterToInference(stream, useEmaForInference);
        }

        /// <summary>
        /// Refreshes ONLY <see cref="InferenceWeights"/> - the post-training refresh
        /// before the composite (SampleRenderer.TrainNrc's tail) needs exactly this;
        /// reconverting <see cref="TrainingWeights"/> there too would be pure waste,
        /// since nothing reads it again until the next trained frame's own full
        /// conversion.
        /// </summary>
        internal void ConvertMasterToInference(AcceleratorStream stream, bool useEmaForInference = false)
        {
            // TrainingOptimal, not InferencingOptimal - the InferencingOptimal curated
            // MatVecMul path was never exercised by any real caller (ForwardKernel had
            // an Inferencing method for it, since removed) and composite/
            // self-training-target read it through the already-validated
            // ForwardKernel.Training path instead once that surfaced a bug. TrainingOptimal
            // is a valid, just not perf-optimal, layout for a forward-only read too.
            //
            // Reads MasterWeights (live) by default, not EmaWeights - matches VkNRC's
            // own default exactly: nrc_optimize.comp's uUseWeights write is gated by a
            // push constant `uUseEMAWeights` that main.cpp initializes false (an
            // unchecked "Use EMA" ImGui debug checkbox, m_use_ema_weights{false} in
            // VkNRCState.hpp) - i.e. both the composite/screen inference AND the
            // self-training bootstrap query (nrc_inference.comp, fed by
            // nrc_resources.use_weights) read the live, continuously-updated weights by
            // default, not an EMA snapshot.
            //
            // useEmaForInference is that VkNRC debug checkbox, now implemented
            // (SampleRenderer.NrcUseEmaInference, UI/UiPanel.cs's "Use EMA weights"
            // checkbox, SweepConfig.UseEmaInference) - it switches ONLY this
            // InferenceWeights conversion to the EMA snapshot; TrainingWeights above
            // always converts from the live master weights, since training itself must
            // never chase its own smoothed shadow. Also the switch that makes
            // NrcEmaAlpha an actually-live parameter - with this false (the default),
            // EMA weights are computed every step but sampled by nothing.
            masterToTrainingOptimal.Convert(
                stream, numNetworks: 1,
                useEmaForInference ? EmaWeights : MasterWeights, inputNetworkStrideInBytes: 0,
                InferenceWeights, outputNetworkStrideInBytes: 0);
        }

        internal void ConvertDwToRowMajor(AcceleratorStream stream)
        {
            trainingOptimalToRowMajor.Convert(
                stream, numNetworks: 1,
                DwConverted, inputNetworkStrideInBytes: 0,
                DwRowMajor, outputNetworkStrideInBytes: 0);
        }

        internal void ZeroDw() => DwConverted.MemSetToZero();

        public void Dispose()
        {
            masterToTrainingOptimal?.Dispose();
            trainingOptimalToRowMajor?.Dispose();
            MasterWeights?.Dispose();
            EmaWeights?.Dispose();
            AdamM?.Dispose();
            AdamV?.Dispose();
            DwRowMajor?.Dispose();
            TrainingWeights?.Dispose();
            DwConverted?.Dispose();
            InferenceWeights?.Dispose();
        }
    }
}
