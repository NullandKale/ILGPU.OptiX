// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: FrameStats.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample13
{
    // Per-frame timing/HUD data, produced once per SampleRenderer.render() call and
    // exposed via SampleRenderer.LastStats. Swapped as one atomic struct assignment
    // rather than several separate properties, so MainWindow's overlay always reads a
    // consistent snapshot instead of a torn mix of two different frames' numbers.
    public struct FrameStats
    {
        public double TraceMs;
        public double DenoiseMs;
        public double TonemapMs;
        // Time to copy the tonemapped frame from GPU to the CPU handoff buffer -
        // includes any wait for the UI thread to finish reading that buffer slot from
        // two frames ago (see SampleRenderer.render()'s bufferFree.Wait()), which is
        // normally near-instant but is exactly the DX12-style backpressure that
        // throttles render() once presentation falls a frame behind.
        public double ReadbackMs;

        // Cost of publishing the buffer (bufferReady.Release()) - just a semaphore
        // release, not a blocking Dispatcher call, so this should be near-zero. See
        // MainWindow's "Display FPS" stat for how fast WPF is actually presenting.
        public double PublishMs;

        // Wall-clock time between the start of this render() call and the start of the
        // previous one (smoothed). Fps = 1000 / TotalFrameMs. Unlike before, this is
        // no longer fully unthrottled - render() can block briefly in ReadbackMs if
        // it's gotten 2 frames ahead of what MainWindow has presented (see
        // SampleRenderer's double-buffered async present handoff), so this and
        // MainWindow's separate "Display FPS" now track each other more closely than
        // they used to under the old fully decoupled scheme.
        public double TotalFrameMs;
        public double Fps;

        public int SamplesAccumulated;
    }
}
