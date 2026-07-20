// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: TrainRecordBuffers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.OptiX.Device;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Sample21.Core;
using System;

namespace Sample21.Rendering
{
    /// <summary>
    /// Owns the per-frame train-record pool and its atomic slot counters, plus the
    /// per-pixel eval-record buffer. <see cref="TrainRecords"/> is one flat buffer of
    /// <see cref="NrcConstants.MaxTrainRecords"/> elements but treated as
    /// <see cref="NrcConstants.TrainBatchesPerFrame"/> disjoint
    /// <see cref="NrcConstants.TrainBatchSize"/>-sized slices (batch <c>b</c>'s records
    /// live at <c>[b*TrainBatchSize, (b+1)*TrainBatchSize)</c>) - matching VkNRC's own
    /// 4-different-batches-per-frame scheme (RaygenProgram.cs randomly assigns each
    /// TrainRecord to one of the 4 batches at write time), not one pool reprocessed 4
    /// times. <see cref="Counter"/> has one atomic slot counter per batch. No
    /// indirect-dispatch infrastructure is needed - every downstream stage launches a
    /// fixed-size grid and bounds-checks against the live device-resident per-batch
    /// counter itself .
    /// </summary>
    public sealed class TrainRecordBuffers : IDisposable
    {
        public MemoryBuffer1D<TrainRecord, Stride1D.Dense> TrainRecords { get; }
        public MemoryBuffer1D<int, Stride1D.Dense> Counter { get; }
        public MemoryBuffer1D<EvalRecord, Stride1D.Dense> EvalRecords { get; private set; }

        public TrainRecordBuffers(CudaAccelerator accelerator)
        {
            TrainRecords = accelerator.Allocate1D<TrainRecord>(NrcConstants.MaxTrainRecords);
            Counter = accelerator.Allocate1D<int>(NrcConstants.TrainBatchesPerFrame);
        }

        /// <summary>Re-sized whenever the render resolution changes - one eval-record slot per pixel.</summary>
        public void ResizeEvalRecords(CudaAccelerator accelerator, int width, int height)
        {
            EvalRecords?.Dispose();
            EvalRecords = accelerator.Allocate1D<EvalRecord>(Math.Max(1, width * height));
        }

        public void ResetCounter(AcceleratorStream stream) => Counter.View.MemSetToZero(stream);

        public OptixDeviceView<TrainRecord> TrainRecordsView => OptixDeviceView<TrainRecord>.From(TrainRecords);
        public OptixDeviceView<EvalRecord> EvalRecordsView => OptixDeviceView<EvalRecord>.From(EvalRecords);
        public OptixDeviceView<int> CounterView => OptixDeviceView<int>.From(Counter);

        public void Dispose()
        {
            TrainRecords?.Dispose();
            Counter?.Dispose();
            EvalRecords?.Dispose();
        }
    }
}
