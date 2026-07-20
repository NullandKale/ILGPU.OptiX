// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: CompositeProgram.cs
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
    /// Forward-only cache query over every active <see cref="EvalRecord"/> (EMA
    /// weights), writing <c>Throughput * cache(handoff)</c> both into the debug
    /// <see cref="NrcPreviewBuffer"/> AND (read-modify-write, <c>.xyz</c> only,
    /// preserving <c>.w</c> = this frame's own view-space depth) added into
    /// <see cref="RawColorBuffer"/> - the cache genuinely replaces the tail bounces
    /// Device/RaygenProgram.cs's adaptive handoff stopped tracing, matching VkNRC's own
    /// architecture. No double-counting: RaygenProgram.cs's bounce loop stops
    /// accumulating radiance at the exact bounce this cache prediction substitutes for
    /// (see its own NRC comment), so RawColorBuffer's existing contents are exactly
    /// "bias" and this addition is exactly "factor * predict".
    /// </summary>
    public struct CompositeLaunchParams
    {
        public OptixDeviceView<EvalRecord> EvalRecords;
        public OptixDeviceView<Vec4> NrcPreviewBuffer;
        public OptixDeviceView<Vec4> RawColorBuffer;
        public int PixelCount;
        // Compositing into RawColorBuffer is only correct when every pixel traced
        // exactly one sample (RaygenProgram.cs's per-pixel NRC state only keeps the
        // LAST sample's snapshot - see its own doc comment) - with NumPixelSamples > 1
        // the eval record represents only 1 of N samples' truncated tail while
        // RawColorBuffer already averages all N, so blending it in at full weight would
        // bias the image. Guarded here rather than silently introducing that bias.
        public int NumPixelSamples;

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

    public static class CompositeProgram
    {
        /// <summary>Generic over the network-width strategy - see GradientProgram.ComputeGradient{TNet}'s own doc comment.</summary>
        public static void Composite<TNet>(CompositeLaunchParams p) where TNet : struct, INrcNetworkOps
        {
            long recordIdx = OptixGetLaunchIndex.X;
            if (recordIdx >= p.PixelCount)
                return;

            EvalRecord record = p.EvalRecords[recordIdx];
            if (record.Active == 0)
            {
                // This pixel's path never handed off (resolved entirely on its own,
                // or NRC is disabled) - RawColorBuffer is already fully/correctly
                // populated by RaygenProgram.cs, so it's left untouched here.
                // NrcPreviewBuffer only exists while the preview is displayed
                // (SampleRenderer allocates it lazily) - invalid view otherwise.
                if (p.NrcPreviewBuffer.IsValid)
                    p.NrcPreviewBuffer[recordIdx] = new Vec4(0f, 0f, 0f, 1f);
                return;
            }

            Vec3 dielectricF0 = new Vec3(0.04f, 0.04f, 0.04f);
            Vec3 specular = dielectricF0 + ((record.Albedo - dielectricF0) * record.Metallic);

            // slotsPerRecord=2: the forward-only ping-pong path (ForwardInference below)
            // - one record per PIXEL here, so the per-record act footprint matters more
            // than anywhere else in the pipeline .
            InputEncoding.Encode(p.ActScratch, recordIdx, record.Position, record.ScatteredDir,
                record.Normal, record.Roughness, record.Albedo, specular, 2, p.LayerWidth);

            // TrainingOptimal weights layout, not Inferencing - see
            // WeightBuffers.ConvertMasterToTrainingAndInference's own note.
            default(TNet).ForwardInference(recordIdx, p.ActScratchBase, p.InferenceWeightsBase, p.HiddenLayerStrideInBytes,
                p.OutputLayerOffsetInBytes, p.ZeroBias64Base, p.ZeroBias3Base, p.ZeroVec64Base, p.OutputScratchBase, p.HiddenLayerCount);

            // Clamped to physical (non-negative) radiance ONLY, no upper bound - see
            // GradientProgram.cs's matching clamp for why (not what VkNRC does).
            Vec3 cachePrediction = new Vec3(
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 0], 0f),
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 1], 0f),
                XMath.Max((float)p.OutputScratch[(recordIdx * 3) + 2], 0f));

            Vec3 result = record.Throughput * cachePrediction;
            if (p.NrcPreviewBuffer.IsValid)
                p.NrcPreviewBuffer[recordIdx] = new Vec4(result.x, result.y, result.z, 1f);

            if (p.NumPixelSamples == 1)
            {
                Vec4 existing = p.RawColorBuffer[recordIdx];
                p.RawColorBuffer[recordIdx] = new Vec4(
                    existing.x + result.x, existing.y + result.y, existing.z + result.z, existing.w);
            }
        }
    }
}
