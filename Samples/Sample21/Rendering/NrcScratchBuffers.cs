// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: NrcScratchBuffers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.OptiX.CooperativeVectors;
using ILGPU.OptiX.Device;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Sample21.Core;
using System;
using Half = ILGPU.Half;

namespace Sample21.Rendering
{
    /// <summary>
    /// Per-record scratch (activations, output, backward-pass temporaries) and the
    /// zero-filled bias/ReLU-zero vectors every forward/backward call needs. Each buffer
    /// is sized for the largest launch that actually indexes it, rather than uniformly
    /// for max(pixel count, MaxTrainRecords):
    /// <list type="bullet">
    /// <item><see cref="ActScratch"/> - dual-sized: the training path (gradient) caches
    /// hiddenLayerCount+1 activation slots per record for up to
    /// <see cref="NrcConstants.MaxTrainRecords"/> records, while the inference paths
    /// (composite at one record per PIXEL, self-training-target) ping-pong just 2 slots
    /// per record (<see cref="Device.Network.ForwardKernel.Inference"/>); whichever
    /// needs more elements wins.</item>
    /// <item><see cref="OutputScratch"/> - indexed per-pixel by the composite, so it
    /// keeps the max(pixels, MaxTrainRecords) capacity.</item>
    /// <item><see cref="DaOutScratch"/>/<see cref="DaScratch"/>/<see cref="DInScratch"/>/
    /// <see cref="MaskScratch"/> - backward-pass only, never indexed past
    /// MaxTrainRecords.</item>
    /// </list>
    /// The three NRC stages run sequentially within a frame, never concurrently, so
    /// sharing one set across them is safe.
    /// </summary>
    public sealed class NrcScratchBuffers : IDisposable
    {
        public MemoryBuffer1D<Half, Stride1D.Dense> ActScratch { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> OutputScratch { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> DaOutScratch { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> DaScratch { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> DInScratch { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> MaskScratch { get; private set; }

        // The property/field names (here and on GradientLaunchParams/etc.'s
        // ZeroBias64Base/ZeroVec64Base) say "64" but don't reflect the actual width -
        // the underlying buffer is resized to whatever LayerWidth is actually active;
        // only functionally 64-wide at the default width.
        public MemoryBuffer1D<Half, Stride1D.Dense> ZeroBias64 { get; private set; }
        public MemoryBuffer1D<Half, Stride1D.Dense> ZeroBias3 { get; }

        private int capacity;
        private int currentHiddenLayerCount = -1;
        private int currentLayerWidth = -1;

        public NrcScratchBuffers(CudaAccelerator accelerator, int hiddenLayerCount = NrcConstants.HiddenLayerCount, int layerWidth = (int)NrcConstants.LayerWidth)
        {
            ZeroBias3 = accelerator.Allocate1D(new Half[NrcConstants.OutputWidth]);
            EnsureCapacity(accelerator, NrcConstants.MaxTrainRecords, hiddenLayerCount, layerWidth);
        }

        /// <summary>
        /// Grow-only on <paramref name="recordCount"/> (the max record count any
        /// inference launch will index - screen pixels for the composite), but ALWAYS
        /// reallocates when <paramref name="hiddenLayerCount"/>/<paramref name="layerWidth"/>
        /// change - called rarely (construction and resize), not per-frame.
        /// </summary>
        public void EnsureCapacity(CudaAccelerator accelerator, int recordCount, int hiddenLayerCount = NrcConstants.HiddenLayerCount, int layerWidth = (int)NrcConstants.LayerWidth)
        {
            if (recordCount <= capacity && hiddenLayerCount == currentHiddenLayerCount && layerWidth == currentLayerWidth)
                return;

            ActScratch?.Dispose();
            OutputScratch?.Dispose();
            DaOutScratch?.Dispose();
            DaScratch?.Dispose();
            DInScratch?.Dispose();
            MaskScratch?.Dispose();

            capacity = Math.Max(recordCount, capacity);
            currentHiddenLayerCount = hiddenLayerCount;
            if (layerWidth != currentLayerWidth)
            {
                ZeroBias64?.Dispose();
                ZeroBias64 = accelerator.Allocate1D(new Half[layerWidth]);
                currentLayerWidth = layerWidth;
            }

            // Dual sizing (this class's own doc comment): training's full per-layer
            // cache over MaxTrainRecords vs inference's 2-slot ping-pong over every
            // pixel - at the default depth the inference term dominates for any
            // window bigger than ~a third of a megapixel, and it no longer scales
            // with network depth at all.
            long trainingActElements = (long)NrcConstants.MaxTrainRecords * (hiddenLayerCount + 1) * layerWidth;
            long inferenceActElements = (long)capacity * 2 * layerWidth;
            ActScratch = accelerator.Allocate1D<Half>(Math.Max(trainingActElements, inferenceActElements));

            OutputScratch = accelerator.Allocate1D<Half>((long)capacity * (int)NrcConstants.OutputWidth);

            // Backward-pass only - nothing ever indexes these past MaxTrainRecords.
            DaOutScratch = accelerator.Allocate1D<Half>((long)NrcConstants.MaxTrainRecords * (int)NrcConstants.OutputWidth);
            DaScratch = accelerator.Allocate1D<Half>((long)NrcConstants.MaxTrainRecords * layerWidth);
            DInScratch = accelerator.Allocate1D<Half>((long)NrcConstants.MaxTrainRecords * layerWidth);
            MaskScratch = accelerator.Allocate1D<Half>((long)NrcConstants.MaxTrainRecords * layerWidth);
        }

        public OptixDeviceView<Half> ActScratchView => OptixDeviceView<Half>.From(ActScratch);
        public OptixDeviceView<Half> OutputScratchView => OptixDeviceView<Half>.From(OutputScratch);
        public OptixDeviceView<Half> DaOutScratchView => OptixDeviceView<Half>.From(DaOutScratch);

        public void Dispose()
        {
            ActScratch?.Dispose();
            OutputScratch?.Dispose();
            DaOutScratch?.Dispose();
            DaScratch?.Dispose();
            DInScratch?.Dispose();
            MaskScratch?.Dispose();
            ZeroBias64?.Dispose();
            ZeroBias3?.Dispose();
        }
    }
}
