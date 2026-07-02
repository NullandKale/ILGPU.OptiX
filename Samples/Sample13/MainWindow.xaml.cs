using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Sample13
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public int width = 1200;
        public int height = 800;
        public WriteableBitmap wBitmap;
        public Int32Rect rect;

        public SampleRenderer sampleRenderer;
        public Thread renderThread;

        public bool run = true;

        // Reader side of SampleRenderer's double-buffered async present handoff (see
        // SampleRenderer.TryPresentFrame/FinishPresent) - this thread's own private
        // "next buffer to read" index, starts at 0 to match resize()'s initial layout.
        // Only ever touched from OnCompositionRendering, i.e. only on the UI thread.
        private int presentIndex;

        // Presentation-side frame pacing, tracked only across ticks where a fresh
        // frame was actually presented (see OnCompositionRendering) - so this reflects
        // real display throughput, not just how often WPF happens to compose.
        private long lastPresentTicks;
        private bool hasPresented;
        private double smoothedPresentMs = 16.0;
        private double displayFps;

        // Mouse-drag camera controls: left = orbit, right = dolly, middle = pan.
        private Point? dragLastPos;
        private MouseButton? dragButton;

        // Scene-cycling debounce, matching the reference's RaytraceEntity 1-second
        // cooldown on its I/U scene-switch keys.
        private DateTime lastSceneSwitch = DateTime.MinValue;
        private static readonly TimeSpan SceneSwitchCooldown = TimeSpan.FromSeconds(1);

        public MainWindow()
        {
            InitializeComponent();

            wBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            rect = new Int32Rect(0, 0, width, height);
            Frame.Source = wBitmap;

            sampleRenderer = new SampleRenderer(width, height, this);
            Closing += MainWindow_Closing;

            Frame.MouseDown += Frame_MouseDown;
            Frame.MouseMove += Frame_MouseMove;
            Frame.MouseUp += Frame_MouseUp;
            Frame.LostMouseCapture += (s, e) => { dragButton = null; };

            KeyDown += MainWindow_KeyDown;

            // CompositionTarget.Rendering fires once per WPF-composed frame (UI thread
            // only, roughly vsync-paced) and tries to pull the next frame the render
            // thread has published (see SampleRenderer.TryPresentFrame) - a DX12-style
            // double-buffered swap chain, not a fully decoupled "latest wins" scheme:
            // the render thread throttles itself against this same handoff (see
            // render()'s bufferFree.Wait()), so the two threads' rates stay coupled
            // within one frame of each other while still overlapping - render can be
            // producing the next frame while this thread is still presenting the
            // previous one. This replaces the old design where the render thread
            // blocked on Dispatcher.Invoke every frame, which tied its throughput
            // directly to the UI thread's own schedule.
            CompositionTarget.Rendering += OnCompositionRendering;

            renderThread = new Thread(renderThreadMain);
            renderThread.Start();
        }

        private void OnCompositionRendering(object sender, EventArgs e)
        {
            if (sampleRenderer.TryPresentFrame(ref presentIndex, out byte[] data))
            {
                long now = Stopwatch.GetTimestamp();
                if (hasPresented)
                {
                    double sincePresentMs = (now - lastPresentTicks) * 1000.0 / Stopwatch.Frequency;
                    smoothedPresentMs = (smoothedPresentMs * 0.9) + (sincePresentMs * 0.1);
                    displayFps = 1000.0 / smoothedPresentMs;
                }
                lastPresentTicks = now;
                hasPresented = true;

                draw(data);
                sampleRenderer.FinishPresent(ref presentIndex);
            }

            updateOverlay();
        }

        // I/U cycle to the next/previous scene, matching the reference's
        // RaytraceEntity.BuildSceneTable() keyboard-cycling UX. D/Space toggles the
        // denoiser and A toggles accumulation, matching Sample11/12's key bindings. M
        // toggles between a single merged triangles GAS build input (fast, default)
        // and one build input per mesh (see SampleRenderer.UseMergedTrianglesGas) -
        // forces a scene rebuild either way, since it changes the GAS/SBT shape.
        private void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.I:
                case Key.U:
                    var now = DateTime.UtcNow;
                    if (now - lastSceneSwitch < SceneSwitchCooldown)
                        return;
                    lastSceneSwitch = now;

                    if (e.Key == Key.I)
                        sampleRenderer.NextScene();
                    else
                        sampleRenderer.PreviousScene();
                    break;

                case Key.D:
                case Key.Space:
                    sampleRenderer.DenoiserOn = !sampleRenderer.DenoiserOn;
                    break;

                case Key.A:
                    sampleRenderer.Accumulate = !sampleRenderer.Accumulate;
                    break;

                case Key.M:
                    sampleRenderer.UseMergedTrianglesGas = !sampleRenderer.UseMergedTrianglesGas;
                    sampleRenderer.RebuildCurrentScene();
                    break;

                case Key.F1:
                    OverlayPanel.Visibility = OverlayPanel.Visibility == Visibility.Visible
                        ? Visibility.Collapsed
                        : Visibility.Visible;
                    return;

                default:
                    return;
            }

            Title = $"ILGPU.Optix Sample13 - {sampleRenderer.CurrentSceneName} - " +
                $"Denoiser {(sampleRenderer.DenoiserOn ? "ON" : "OFF")}, Accumulate {(sampleRenderer.Accumulate ? "ON" : "OFF")}, " +
                $"TrianglesGAS {(sampleRenderer.UseMergedTrianglesGas ? "merged" : "per-mesh")}";
        }

        private void Frame_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton != MouseButton.Left &&
                e.ChangedButton != MouseButton.Right &&
                e.ChangedButton != MouseButton.Middle)
            {
                return;
            }

            dragButton = e.ChangedButton;
            dragLastPos = e.GetPosition(Frame);
            Frame.CaptureMouse();
        }

        private void Frame_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragButton == null || dragLastPos == null)
            {
                return;
            }

            Point pos = e.GetPosition(Frame);
            float dx = (float)((pos.X - dragLastPos.Value.X) / Frame.ActualWidth);
            float dy = (float)((pos.Y - dragLastPos.Value.Y) / Frame.ActualHeight);
            dragLastPos = pos;

            Camera current = sampleRenderer.camera;
            Camera updated = dragButton switch
            {
                MouseButton.Left => CameraMotion.Orbit(current, dx, dy),
                MouseButton.Right => CameraMotion.Dolly(current, dy),
                MouseButton.Middle => CameraMotion.Pan(current, dx, dy),
                _ => current
            };
            sampleRenderer.setCamera(updated);
        }

        private void Frame_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == dragButton)
            {
                dragButton = null;
                Frame.ReleaseMouseCapture();
            }
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            run = false;
            CompositionTarget.Rendering -= OnCompositionRendering;
            // render() notices run==false within its bufferFree wait loop's 50ms
            // retry granularity (see SampleRenderer.render()), so this returns quickly
            // - waiting for it here (rather than disposing GPU resources out from
            // under a still-running render() call) is what actually makes shutdown
            // safe, not just fast.
            renderThread.Join();
            sampleRenderer.Dispose();
        }

        public void renderThreadMain()
        {
            while (run)
            {
                sampleRenderer.render();
            }
        }

        public void draw(byte[] data)
        {
            if (data.Length == wBitmap.PixelWidth * wBitmap.PixelHeight * 4)
            {
                wBitmap.Lock();
                IntPtr pBackBuffer = wBitmap.BackBuffer;
                Marshal.Copy(data, 0, pBackBuffer, data.Length);
                wBitmap.AddDirtyRect(rect);
                wBitmap.Unlock();
            }
        }

        // Called once per composed frame from OnCompositionRendering, whether or not a
        // fresh frame was actually presented this tick - it just reads whatever
        // SampleRenderer.LastStats currently holds, same as before.
        // F1 toggles OverlayPanel.Visibility; skip the formatting work entirely while
        // it's hidden.
        public void updateOverlay()
        {
            if (OverlayPanel.Visibility != Visibility.Visible)
                return;

            var r = sampleRenderer;
            var stats = r.LastStats;

            string primitiveCounts =
                $"Sph {r.SphereCount}  Box {r.BoxCount}\n" +
                $"CylY {r.CylinderYCount}  Disk {r.DiskCount}\n" +
                $"XYRect {r.XYRectCount}  XZRect {r.XZRectCount}\n" +
                $"YZRect {r.YZRectCount}" +
                (r.HasVolumeGrid ? $"\nGrid {r.VolumeGridDims.x}x{r.VolumeGridDims.y}x{r.VolumeGridDims.z}" : "");

            OverlayText.Text =
                $"{r.CurrentSceneName}\n" +
                $"Tris {r.TriangleCount}  Mats {r.MaterialCount}\n" +
                $"{primitiveCounts}\n" +
                $"{r.AccelStructureSummary}\n" +
                $"Render FPS {stats.Fps:0.0}\n" +
                $"Display FPS {displayFps:0.0}\n" +
                $"Frame {stats.TotalFrameMs:0.00} ms\n" +
                $"Samples {stats.SamplesAccumulated}\n" +
                $"Trace {stats.TraceMs:0.00} ms\n" +
                $"Denoise {stats.DenoiseMs:0.00} ms\n" +
                $"Tonemap {stats.TonemapMs:0.00} ms\n" +
                $"Readback {stats.ReadbackMs:0.00} ms\n" +
                $"Publish {stats.PublishMs:0.00} ms\n" +
                $"Denoiser {(r.DenoiserOn ? "ON" : "OFF")}\n" +
                $"Accumulate {(r.Accumulate ? "ON" : "OFF")}";
        }
    }
}
