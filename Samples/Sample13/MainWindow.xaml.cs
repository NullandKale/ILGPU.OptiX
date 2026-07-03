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

        // FPS-style camera controls: WASD for movement, mouse for look
        private bool keyW, keyA, keyS, keyD;
        private Point lastMousePos;
        private float cameraYaw = 0f;
        private float cameraPitch = 0f;
        private float mouseDeltaX = 0f;
        private float mouseDeltaY = 0f;
        private const float MouseSensitivity = 0.1f;
        // Fraction of the scene's CameraWorldScale moved per frame while a WASD key is
        // held - an absolute constant here would be imperceptible in a large scene
        // (e.g. Sponza's worldScale is in the thousands) and far too fast in a small
        // one. 0.01 matches the old fixed 0.1f constant's feel for a worldScale of 10,
        // the typical small-scene value.
        private const float MoveSpeedFraction = 0.01f;

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

            KeyDown += MainWindow_KeyDown;
            KeyUp += MainWindow_KeyUp;

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
            // Apply accumulated mouse delta
            if (mouseDeltaX != 0 || mouseDeltaY != 0)
            {
                cameraYaw += mouseDeltaX;
                cameraPitch += mouseDeltaY;
                cameraPitch = System.Math.Max(-89f, System.Math.Min(89f, cameraPitch));
                mouseDeltaX = 0;
                mouseDeltaY = 0;
                UpdateFPSCamera();
            }

            // Handle FPS movement input
            if (keyW || keyA || keyS || keyD)
            {
                UpdateFPSMovement();
            }

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
            // Track WASD for FPS movement
            switch (e.Key)
            {
                case Key.W: keyW = true; e.Handled = true; return;
                case Key.A: keyA = true; e.Handled = true; return;
                case Key.S: keyS = true; e.Handled = true; return;
                case Key.D: keyD = true; e.Handled = true; return;

                case Key.M:
                    sampleRenderer.UseMergedTrianglesGas = !sampleRenderer.UseMergedTrianglesGas;
                    sampleRenderer.RebuildCurrentScene();
                    return;

                case Key.F1:
                    var controlPanel = this.FindName("ControlPanel") as Border;
                    if (controlPanel != null)
                    {
                        controlPanel.Visibility = controlPanel.Visibility == Visibility.Visible
                            ? Visibility.Collapsed
                            : Visibility.Visible;
                    }
                    return;

                default:
                    return;
            }
        }

        private void MainWindow_KeyUp(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.W: keyW = false; e.Handled = true; break;
                case Key.A: keyA = false; e.Handled = true; break;
                case Key.S: keyS = false; e.Handled = true; break;
                case Key.D: keyD = false; e.Handled = true; break;
            }
        }

        private bool isMouseCaptured = false;

        private void Frame_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton != MouseButton.Left)
            {
                return;
            }

            // Initialize camera angles from current look direction
            Camera current = sampleRenderer.camera;
            Vec3 forward = Vec3.unitVector(current.lookAt - current.origin);

            cameraYaw = (float)System.Math.Atan2(forward.x, forward.z) * 180f / (float)System.Math.PI;
            float verticalAngle = (float)System.Math.Asin(System.Math.Max(-1, System.Math.Min(1, forward.y)));
            cameraPitch = verticalAngle * 180f / (float)System.Math.PI;

            // Capture mouse for FPS camera control
            isMouseCaptured = true;
            lastMousePos = e.GetPosition(this);
            Frame.Focus();
            Mouse.Capture(Frame);
            e.Handled = true;
        }

        private void Frame_MouseMove(object sender, MouseEventArgs e)
        {
            // Only process mouse movement if mouse is captured (left button held)
            if (!isMouseCaptured)
            {
                return;
            }

            Point pos = e.GetPosition(this);
            mouseDeltaX += (float)(pos.X - lastMousePos.X) * MouseSensitivity;
            mouseDeltaY -= (float)(pos.Y - lastMousePos.Y) * MouseSensitivity;
            lastMousePos = pos;
        }

        private void Frame_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left && isMouseCaptured)
            {
                isMouseCaptured = false;
                Mouse.Capture(null);
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
            var r = sampleRenderer;
            var stats = r.LastStats;

            string statsText =
                $"Scene: {r.CurrentSceneName}\n" +
                $"Geometry: {r.TriangleCount} tris, {r.MaterialCount} mats\n" +
                $"\n" +
                $"Render: {stats.Fps:0.0} fps ({stats.TotalFrameMs:0.00} ms)\n" +
                $"Display: {displayFps:0.0} fps\n" +
                $"Samples: {stats.SamplesAccumulated}\n" +
                $"\n" +
                $"Trace:    {stats.TraceMs:0.00} ms\n" +
                $"Denoise:  {stats.DenoiseMs:0.00} ms\n" +
                $"Tonemap:  {stats.TonemapMs:0.00} ms\n" +
                $"Readback: {stats.ReadbackMs:0.00} ms\n" +
                $"Publish:  {stats.PublishMs:0.00} ms\n" +
                $"\n" +
                $"Denoiser: {(r.DenoiserOn ? "ON" : "OFF")}\n" +
                $"Accumulate: {(r.Accumulate ? "ON" : "OFF")}";

            OverlayText.Text = statsText;
            UpdateBounceCountDisplay();
        }

        private void UpdateBounceCountDisplay()
        {
            MirrorCountText.Text = sampleRenderer.MaxMirrorBounces.ToString();
            RefractionCountText.Text = sampleRenderer.MaxRefractionBounces.ToString();
            DiffuseCountText.Text = sampleRenderer.MaxDiffuseBounces.ToString();
            SceneNameText.Text = sampleRenderer.CurrentSceneName;
            DenoiserStatusText.Text = sampleRenderer.DenoiserOn ? "ON" : "OFF";
            AccumulateStatusText.Text = sampleRenderer.Accumulate ? "ON" : "OFF";
        }

        private void PrevSceneBtn_Click(object sender, RoutedEventArgs e)
        {
            var now = DateTime.UtcNow;
            if (now - lastSceneSwitch < SceneSwitchCooldown)
                return;
            lastSceneSwitch = now;
            sampleRenderer.PreviousScene();
            UpdateBounceCountDisplay();
        }

        private void NextSceneBtn_Click(object sender, RoutedEventArgs e)
        {
            var now = DateTime.UtcNow;
            if (now - lastSceneSwitch < SceneSwitchCooldown)
                return;
            lastSceneSwitch = now;
            sampleRenderer.NextScene();
            UpdateBounceCountDisplay();
        }

        private void DenoiserBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.DenoiserOn = !sampleRenderer.DenoiserOn;
            UpdateBounceCountDisplay();
        }

        private void AccumulateBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.Accumulate = !sampleRenderer.Accumulate;
            UpdateBounceCountDisplay();
        }

        private void MirrorIncBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxMirrorBounces = System.Math.Min(8, sampleRenderer.MaxMirrorBounces + 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void MirrorDecBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxMirrorBounces = System.Math.Max(0, sampleRenderer.MaxMirrorBounces - 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void RefractionIncBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxRefractionBounces = System.Math.Min(8, sampleRenderer.MaxRefractionBounces + 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void RefractionDecBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxRefractionBounces = System.Math.Max(0, sampleRenderer.MaxRefractionBounces - 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void DiffuseIncBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxDiffuseBounces = System.Math.Min(8, sampleRenderer.MaxDiffuseBounces + 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void DiffuseDecBtn_Click(object sender, RoutedEventArgs e)
        {
            sampleRenderer.MaxDiffuseBounces = System.Math.Max(0, sampleRenderer.MaxDiffuseBounces - 1);
            sampleRenderer.setCamera(sampleRenderer.camera);
            UpdateBounceCountDisplay();
        }

        private void UpdateFPSMovement()
        {
            Camera current = sampleRenderer.camera;
            Vec3 forward = Vec3.unitVector(current.lookAt - current.origin);
            // Must match Camera.axis.x's own convention (OrthoNormalBasis.fromZY:
            // cross(up, forward), not cross(forward, up)) - axis.x is what actually
            // determines which world direction appears on the right side of the
            // rendered image, so using the opposite cross-product order here silently
            // swaps A/D.
            Vec3 right = Vec3.unitVector(Vec3.cross(current.up, forward));
            Vec3 worldUp = Vec3.unitVector(current.up);

            float moveSpeed = current.worldScale * MoveSpeedFraction;
            Vec3 movement = new Vec3(0, 0, 0);
            if (keyW) movement += forward * moveSpeed;
            if (keyS) movement -= forward * moveSpeed;
            if (keyD) movement += right * moveSpeed;
            if (keyA) movement -= right * moveSpeed;

            Vec3 newOrigin = current.origin + movement;
            Camera updated = new Camera(newOrigin, current.lookAt + movement, current.up, width, height, current.noHitColor, current.verticalFov, current.worldScale);
            sampleRenderer.setCamera(updated);
        }

        private void UpdateFPSCamera()
        {
            Camera current = sampleRenderer.camera;
            float yawRad = cameraYaw * (float)System.Math.PI / 180f;
            float pitchRad = cameraPitch * (float)System.Math.PI / 180f;

            float forwardX = (float)System.Math.Sin(yawRad) * (float)System.Math.Cos(pitchRad);
            float forwardY = (float)System.Math.Sin(pitchRad);
            float forwardZ = (float)System.Math.Cos(yawRad) * (float)System.Math.Cos(pitchRad);
            Vec3 forward = Vec3.unitVector(new Vec3(forwardX, forwardY, forwardZ));

            Vec3 lookAt = current.origin + forward * 1.0f;
            Camera updated = new Camera(current.origin, lookAt, new Vec3(0, 1, 0), width, height, current.noHitColor, current.verticalFov, current.worldScale);
            sampleRenderer.setCamera(updated);
        }
    }
}
