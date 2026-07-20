namespace Sample13
{
    // Per-frame timing/HUD data, produced once per SampleRenderer.render() call and
    // exposed via SampleRenderer.LastStats - logged to the console each frame (no
    // in-viewport text renderer yet). DenoiseAndTonemap's Map/tonemap/Unmap sequence is
    // the last GPU-touching step and its cost is already folded into TonemapMs.
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
    }
}
