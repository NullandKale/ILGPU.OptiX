using ImGuiNET;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using Sample15.UI;
using Sample15.UI.Backends;
using System;

namespace Sample15
{
    /// <summary>
    /// M3: FPS camera (WASD + hold-left-mouse-drag look) and the keyboard-only control
    /// scheme replacing Sample13's WPF button panel (see docs/SAMPLE14_PLAN.md's
    /// Window/input-layer section for the full keymap). M7 adds an ImGui-based visual
    /// panel (UI/UiPanel.cs) mirroring the same controls - the keymap stays as the
    /// keyboard-only fallback it always was, ImGui is an additional, not replacement,
    /// input path. GL context and CUDA/OptiX compute run on the same thread, driven by
    /// OnRenderFrame/OnUpdateFrame - see the plan's threading-model section.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        SampleRenderer sampleRenderer;
        FullscreenQuad quad;
        int frameCount;

        // FPS camera state - same math as Sample13's MainWindow.xaml.cs
        // UpdateFPSMovement/UpdateFPSCamera (written entirely against the UI-framework-
        // agnostic Camera/Vec3/CameraMotion types, so it ports with no math changes,
        // only the input-source glue below). Mouse delta is computed manually from
        // absolute MouseState.X/Y (no MouseState.Delta convenience property exists in
        // OpenTK 4.8 - same pattern example/OpenTKSplat/OpenTKSplat/Graphics/Camera.cs
        // uses).
        float cameraYaw;
        float cameraPitch;
        bool wasLeftMouseDown;
        float lastMouseX;
        float lastMouseY;

        const float MouseSensitivity = 0.1f;
        const float MoveSpeedFraction = 0.01f;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.1f, 0.15f, 0.2f, 1.0f);

            // SampleRenderer's ctor creates the CudaAccelerator and the CUDA-GL interop
            // buffer, both of which need their respective contexts current on this
            // thread - base.OnLoad() already made the GL context current.
            sampleRenderer = new SampleRenderer(ClientSize.X, ClientSize.Y);
            quad = new FullscreenQuad();

            // ImGui.CreateContext() must run before either backend Init() call, and
            // the platform backend (input) must be initialized before the renderer
            // backend - see docs/SAMPLE14_PLAN.md's M7 milestone. Docking/multi-
            // viewport support (ImGuiConfigFlags.DockingEnable/ViewportsEnable) is
            // deliberately not enabled - Sample14 only needs one overlay panel, and
            // multi-viewport requires OpenTK APIs (window.MousePassthrough, several
            // MouseCursor shapes) not present in the OpenTK 4.8.2 version referenced
            // here (see UI/Backends/ImguiImplOpenTK4.cs's patches).
            ImGui.CreateContext();
            ImGuiIOPtr io = ImGui.GetIO();
            io.ConfigFlags |= ImGuiConfigFlags.NavEnableKeyboard;
            ImGui.StyleColorsDark();
            ImguiImplOpenTK4.Init(this);
            ImguiImplOpenGL3.Init();

            PrintControls();
        }

        static void PrintControls()
        {
            Console.WriteLine("[Controls] W/A/S/D move, hold Left Mouse to look, [/] prev/next scene,");
            Console.WriteLine("[Controls] M toggle merged-GAS, Space denoiser, Tab accumulate, V TAA,");
            Console.WriteLine("[Controls] R auto render-scale, 1/2 bounces -/+, T tonemap operator, Esc quit");
        }

        protected override void OnUpdateFrame(FrameEventArgs args)
        {
            base.OnUpdateFrame(args);

            if (KeyboardState.IsKeyDown(Keys.Escape))
            {
                Close();
                return;
            }

            // Don't let camera/scene keys and mouse-look fight with ImGui panel
            // interaction - e.g. clicking a button in the panel shouldn't also start a
            // mouse-look drag underneath it, and typing shouldn't be needed here since
            // there are no text fields, but the keyboard check is symmetric with the
            // mouse one for the same reason.
            var io = ImGui.GetIO();
            if (!io.WantCaptureMouse)
                UpdateMouseLook();
            if (!io.WantCaptureKeyboard)
            {
                UpdateWasdMovement();
                UpdateOneShotKeys();
            }
        }

        void UpdateMouseLook()
        {
            bool leftDown = MouseState.IsButtonDown(MouseButton.Left);
            float x = MouseState.X;
            float y = MouseState.Y;

            if (leftDown && !wasLeftMouseDown)
            {
                // Just pressed - initialize yaw/pitch from the camera's current look
                // direction (so the view doesn't snap on the first drag frame) and
                // reset the delta baseline, mirroring Sample13's Frame_MouseDown fix
                // for the "lookAt flips on first click" bug.
                Camera current = sampleRenderer.camera;
                Vec3 forward = Vec3.unitVector(current.lookAt - current.origin);
                cameraYaw = MathF.Atan2(forward.x, forward.z) * 180f / MathF.PI;
                cameraPitch = MathF.Asin(Math.Clamp(forward.y, -1f, 1f)) * 180f / MathF.PI;
                lastMouseX = x;
                lastMouseY = y;
            }
            else if (leftDown)
            {
                float dx = (x - lastMouseX) * MouseSensitivity;
                float dy = (y - lastMouseY) * MouseSensitivity;
                lastMouseX = x;
                lastMouseY = y;

                cameraYaw += dx;
                cameraPitch = Math.Clamp(cameraPitch - dy, -89f, 89f);
                ApplyCameraLook();
            }

            wasLeftMouseDown = leftDown;
        }

        void ApplyCameraLook()
        {
            Camera current = sampleRenderer.camera;
            float yawRad = cameraYaw * MathF.PI / 180f;
            float pitchRad = cameraPitch * MathF.PI / 180f;

            float forwardX = MathF.Sin(yawRad) * MathF.Cos(pitchRad);
            float forwardY = MathF.Sin(pitchRad);
            float forwardZ = MathF.Cos(yawRad) * MathF.Cos(pitchRad);
            Vec3 forward = Vec3.unitVector(new Vec3(forwardX, forwardY, forwardZ));

            Vec3 lookAt = current.origin + (forward * 1.0f);
            Camera updated = new Camera(current.origin, lookAt, new Vec3(0f, 1f, 0f), ClientSize.X, ClientSize.Y,
                current.noHitColor, current.verticalFov, current.worldScale);
            sampleRenderer.setCamera(updated);
        }

        void UpdateWasdMovement()
        {
            bool w = KeyboardState.IsKeyDown(Keys.W);
            bool a = KeyboardState.IsKeyDown(Keys.A);
            bool s = KeyboardState.IsKeyDown(Keys.S);
            bool d = KeyboardState.IsKeyDown(Keys.D);
            if (!(w || a || s || d))
                return;

            Camera current = sampleRenderer.camera;
            Vec3 forward = Vec3.unitVector(current.lookAt - current.origin);
            // Must match Camera.axis.x's own convention (OrthoNormalBasis.fromZY:
            // cross(up, forward), not cross(forward, up)) - axis.x is what actually
            // determines which world direction appears on the right side of the
            // rendered image, so using the opposite cross-product order here silently
            // swaps A/D.
            Vec3 right = Vec3.unitVector(Vec3.cross(current.up, forward));

            float moveSpeed = current.worldScale * MoveSpeedFraction;
            Vec3 movement = new Vec3(0f, 0f, 0f);
            if (w) movement += forward * moveSpeed;
            if (s) movement -= forward * moveSpeed;
            if (d) movement += right * moveSpeed;
            if (a) movement -= right * moveSpeed;

            Vec3 newOrigin = current.origin + movement;
            Camera updated = new Camera(newOrigin, current.lookAt + movement, current.up, ClientSize.X, ClientSize.Y,
                current.noHitColor, current.verticalFov, current.worldScale);
            sampleRenderer.setCamera(updated);
        }

        void UpdateOneShotKeys()
        {
            if (IsKeyPressed(Keys.LeftBracket))
            {
                sampleRenderer.PreviousScene();
                Console.WriteLine($"[Scene] {sampleRenderer.CurrentSceneName}");
            }
            if (IsKeyPressed(Keys.RightBracket))
            {
                sampleRenderer.NextScene();
                Console.WriteLine($"[Scene] {sampleRenderer.CurrentSceneName}");
            }

            if (IsKeyPressed(Keys.M))
            {
                sampleRenderer.UseMergedTrianglesGas = !sampleRenderer.UseMergedTrianglesGas;
                sampleRenderer.RebuildCurrentScene();
                Console.WriteLine($"[MergedGAS] {sampleRenderer.UseMergedTrianglesGas}");
            }

            if (IsKeyPressed(Keys.Space))
            {
                sampleRenderer.DenoiserOn = !sampleRenderer.DenoiserOn;
                Console.WriteLine($"[Denoiser] {sampleRenderer.DenoiserOn}");
            }
            if (IsKeyPressed(Keys.Tab))
            {
                sampleRenderer.Accumulate = !sampleRenderer.Accumulate;
                Console.WriteLine($"[Accumulate] {sampleRenderer.Accumulate}");
            }
            if (IsKeyPressed(Keys.V))
            {
                sampleRenderer.TemporalDenoiseEnabled = !sampleRenderer.TemporalDenoiseEnabled;
                Console.WriteLine($"[TAA] {sampleRenderer.TemporalDenoiseEnabled}");
            }
            if (IsKeyPressed(Keys.R))
            {
                sampleRenderer.AutoScaleEnabled = !sampleRenderer.AutoScaleEnabled;
                Console.WriteLine($"[AutoScale] {sampleRenderer.AutoScaleEnabled}");
            }

            if (IsKeyPressed(Keys.D1)) AdjustMaxBounces(-1);
            if (IsKeyPressed(Keys.D2)) AdjustMaxBounces(1);

            // Tonemap controls (docs/SAMPLE15_PLAN.md Milestone M8) - Exposure/env
            // rotation/env intensity are continuous slider values with no natural
            // keyboard equivalent (unlike the discrete toggles here), so they stay
            // UI-panel-only.
            if (IsKeyPressed(Keys.T))
            {
                sampleRenderer.TonemapOperator = sampleRenderer.TonemapOperator == TonemapKernel.OperatorReinhard
                    ? TonemapKernel.OperatorAces
                    : TonemapKernel.OperatorReinhard;
                Console.WriteLine($"[Tonemap] {(sampleRenderer.TonemapOperator == TonemapKernel.OperatorAces ? "ACES" : "Reinhard")}");
            }
        }

        // Unified bounce budget (docs/SAMPLE15_PLAN.md Milestone M3) - replaces the
        // old three per-material-kind Adjust*Bounces methods/key bindings (D1-D6) now
        // that every material kind shares one raygen bounce loop terminated by
        // Russian roulette; D3-D6 are free again.
        void AdjustMaxBounces(int delta)
        {
            sampleRenderer.MaxBounces = Math.Clamp(sampleRenderer.MaxBounces + delta, 1, 32);
            sampleRenderer.setCamera(sampleRenderer.camera);
            Console.WriteLine($"[MaxBounces] {sampleRenderer.MaxBounces}");
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            sampleRenderer.render();
            sampleRenderer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(sampleRenderer.GlTextureHandle);

            // ImGui draws on top of the already-blitted scene quad, same framebuffer,
            // same frame - NewFrame (both backends) -> build widgets -> Render ->
            // RenderDrawData, matching example/ImGui.NET_OpenTK_Sample's own sequence.
            ImguiImplOpenGL3.NewFrame();
            ImguiImplOpenTK4.NewFrame();
            ImGui.NewFrame();

            UiPanel.Draw(sampleRenderer);

            ImGui.Render();
            ImguiImplOpenGL3.RenderDrawData(ImGui.GetDrawData());

            SwapBuffers();

            // Console stats logging (see docs/SAMPLE14_PLAN.md) stays in place
            // alongside the panel's own STATS section - useful when the panel is
            // hidden/off-screen or for scripted/headless runs.
            frameCount++;
            if (frameCount % 120 == 0)
            {
                var stats = sampleRenderer.LastStats;
                Console.WriteLine(
                    $"[Stats] Scene: {sampleRenderer.CurrentSceneName} | " +
                    $"{stats.Fps:0.0} fps ({stats.TotalFrameMs:0.00} ms) | " +
                    $"Samples: {stats.SamplesAccumulated} | " +
                    $"Trace: {stats.TraceMs:0.00}ms Denoise: {stats.DenoiseMs:0.00}ms Tonemap: {stats.TonemapMs:0.00}ms");
            }
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            // GL viewport always matches the actual window/backbuffer size - the
            // fullscreen quad's textured draw (FullscreenQuad.Draw) already stretches
            // whatever resolution the render texture is (linear-filtered) to fill
            // whatever viewport is set, so a lower-than-window render resolution
            // (SampleRenderer.RenderScale) just softens the image, it never
            // letterboxes or distorts it.
            GL.Viewport(0, 0, e.Width, e.Height);

            // OpenTK can fire an initial resize before OnLoad() has constructed
            // sampleRenderer yet.
            sampleRenderer?.OnWindowResized(e.Width, e.Height);
        }

        protected override void OnUnload()
        {
            ImguiImplOpenGL3.Shutdown();
            ImguiImplOpenTK4.Shutdown();
            quad?.Dispose();
            sampleRenderer?.Dispose();
            base.OnUnload();
        }
    }
}
