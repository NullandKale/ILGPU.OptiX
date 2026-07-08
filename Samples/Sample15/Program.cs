using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using System;

namespace Sample15
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // --bench: run a fixed, reproducible Sponza workload for
            // BenchmarkRunner.FrameCount frames, print the average frame time, and
            // exit - see BenchmarkRunner's own doc comment for why (A/B-testing
            // rendering-path changes against a number that isn't just the live,
            // vsync/auto-scale-influenced FPS counter).
            bool benchMode = Array.IndexOf(args, "--bench") >= 0;

            var gameWindowSettings = GameWindowSettings.Default;
            var nativeWindowSettings = new NativeWindowSettings
            {
                ClientSize = new Vector2i(1200, 800),
                Title = "Sample15 - PBR Path Tracer",
                Flags = ContextFlags.ForwardCompatible,
                Profile = ContextProfile.Core,
                APIVersion = new System.Version(3, 3),
            };

            using var window = new RenderWindow(gameWindowSettings, nativeWindowSettings, benchMode);
            window.Run();
        }
    }
}
