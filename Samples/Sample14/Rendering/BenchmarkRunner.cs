using System;
using System.Diagnostics;

namespace Sample14
{
    /// <summary>
    /// Performance benchmark mode (the <c>--bench</c> command-line flag - see
    /// Program.cs/RenderWindow.cs). Switches to the Sponza scene, disables auto-scale
    /// (the render resolution must stay fixed for a repeatable measurement - see
    /// SampleRenderer.AutoScaleEnabled's own doc comment on why it resizes mid-session
    /// otherwise) and VSync (which would otherwise cap wall-clock frame time at the
    /// monitor refresh rate and hide any actual GPU-side speedup/regression), then
    /// times exactly <see cref="FrameCount"/> consecutive SampleRenderer.render() calls
    /// and prints the average wall-clock frame time/FPS plus the averaged GPU-side
    /// trace/denoise/tonemap breakdown before the caller closes the window. Intended
    /// for A/B-testing rendering-path changes (e.g. the Device/ kernel optimization
    /// passes) against a fixed, reproducible workload instead of eyeballing the live
    /// FPS counter.
    /// </summary>
    public sealed class BenchmarkRunner
    {
        public const int FrameCount = 500;
        const int ProgressPrintIntervalFrames = 100;

        readonly Stopwatch stopwatch = new Stopwatch();
        int framesTimed;
        double traceMsSum;
        double denoiseMsSum;
        double tonemapMsSum;
        bool aborted;

        public bool Finished { get; private set; }

        /// <summary>
        /// Switches to the Sponza scene and applies the fixed-workload settings the
        /// benchmark measures under. Call once, right after the renderer/window are
        /// constructed and before the first OnFrameRendered call.
        /// </summary>
        public void Start(SampleRenderer renderer)
        {
            int sponzaIndex = renderer.FindSceneIndexByBuilderName(nameof(MeshScenes.BuildSponzaScene));
            if (sponzaIndex < 0)
            {
                Console.WriteLine("[Bench] Could not find the Sponza scene in the scene roster - aborting.");
                aborted = true;
                Finished = true;
                return;
            }

            renderer.SwitchToScene(sponzaIndex);
            renderer.AutoScaleEnabled = false;

            Console.WriteLine($"[Bench] Running {FrameCount} frames of '{renderer.CurrentSceneName}' " +
                $"at {renderer.RenderWidth}x{renderer.RenderHeight}, auto-scale off...");
        }

        /// <summary>
        /// Call once per rendered frame (after SampleRenderer.render() has updated
        /// LastStats). No-ops once <see cref="Finished"/> is true - the caller is
        /// expected to close the window as soon as that happens.
        /// </summary>
        public void OnFrameRendered(SampleRenderer renderer)
        {
            if (Finished || aborted)
                return;

            if (!stopwatch.IsRunning)
                stopwatch.Start();

            FrameStats stats = renderer.LastStats;
            traceMsSum += stats.TraceMs;
            denoiseMsSum += stats.DenoiseMs;
            tonemapMsSum += stats.TonemapMs;
            framesTimed++;

            if (framesTimed % ProgressPrintIntervalFrames == 0 && framesTimed < FrameCount)
                Console.WriteLine($"[Bench] {framesTimed}/{FrameCount} frames...");

            if (framesTimed >= FrameCount)
            {
                stopwatch.Stop();
                Report();
                Finished = true;
            }
        }

        void Report()
        {
            double avgFrameMs = stopwatch.Elapsed.TotalMilliseconds / framesTimed;
            double avgFps = 1000.0 / avgFrameMs;
            double avgTraceMs = traceMsSum / framesTimed;
            double avgDenoiseMs = denoiseMsSum / framesTimed;
            double avgTonemapMs = tonemapMsSum / framesTimed;

            Console.WriteLine("[Bench] ==================== Results ====================");
            Console.WriteLine($"[Bench] Frames:            {framesTimed}");
            Console.WriteLine($"[Bench] Avg frame time:    {avgFrameMs:0.000} ms  ({avgFps:0.0} fps)");
            Console.WriteLine($"[Bench] Avg trace:         {avgTraceMs:0.000} ms");
            Console.WriteLine($"[Bench] Avg denoise:       {avgDenoiseMs:0.000} ms");
            Console.WriteLine($"[Bench] Avg tonemap:       {avgTonemapMs:0.000} ms");
            Console.WriteLine("[Bench] ==================================================");
        }
    }
}
