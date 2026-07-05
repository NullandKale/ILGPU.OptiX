namespace Sample15
{
    // Per-frame timing/HUD data, produced once per SampleRenderer.render() call and
    // exposed via SampleRenderer.LastStats - logged to the console each frame (no
    // in-viewport text renderer yet, see docs/SAMPLE14_PLAN.md's HUD deferral note).
    //
    // No ReadbackMs/PublishMs here unlike Sample13's version: those measured the CPU
    // readback + double-buffered handoff to a WPF UI thread, neither of which exist in
    // Sample14 - DenoiseAndTonemap's Map/tonemap/Unmap sequence is the last GPU-touching
    // step and its cost is already folded into TonemapMs.
    public struct FrameStats
    {
        public double TraceMs;
        public double DenoiseMs;
        public double TonemapMs;

        // Wall-clock time between the start of this render() call and the start of the
        // previous one (smoothed). Fps = 1000 / TotalFrameMs.
        public double TotalFrameMs;
        public double Fps;

        public int SamplesAccumulated;

        // Primary-ray throughput this frame (renderWidth*renderHeight*NumPixelSamples
        // / TraceMs) - the control signal for SampleRenderer's render-scale slider/
        // auto-scaler and its own UI readout, not a full "every ray including bounces/
        // shadows" count (Russian roulette makes that path-length-dependent and not
        // worth instrumenting just for a resolution-scaling control signal).
        public double RaysPerSecond;
    }
}
