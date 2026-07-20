namespace Sample21
{
    // Per-frame timing/HUD data, produced once per SampleRenderer.render() call and
    // exposed via SampleRenderer.LastStats - logged to the console each frame (no
    // in-viewport text renderer yet). No CPU readback stage: DenoiseAndTonemap's
    // Map/tonemap/Unmap sequence is the last GPU-touching step and its cost is already
    // folded into TonemapMs.
    public struct FrameStats
    {
        public double TraceMs;
        // The whole NRC side channel's GPU cost this frame (self-training-target +
        // gradient/optimize batches + composite, SampleRenderer.RunNrcTrainingAndComposite)
        // - 0 when NRC is disabled. Split out from TraceMs so performance measurements
        // can attribute cost to the cache itself rather than the path trace.
        public double NrcMs;
        public double DenoiseMs;
        public double TonemapMs;

        // Wall-clock time between the start of this render() call and the start of the
        // previous one (smoothed). Fps = 1000 / TotalFrameMs.
        public double TotalFrameMs;
        public double Fps;

        public int SamplesAccumulated;
    }
}
