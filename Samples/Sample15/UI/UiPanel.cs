using ImGuiNET;
using System.Numerics;

namespace Sample15.UI
{
    /// <summary>
    /// The ImGui control panel replacing Sample13's WPF button panel (see
    /// docs/SAMPLE14_PLAN.md's M7 milestone) - same four sections (SCENE, RENDER
    /// SETTINGS, OPTIONS, CONTROLS) plus a stats readout, laid out top-left over the
    /// rendered frame. Every control here has a keyboard equivalent too (see
    /// RenderWindow's UpdateOneShotKeys) - this panel is a visual/mouse-driven
    /// alternative, not a replacement for the keymap.
    /// </summary>
    public static class UiPanel
    {
        public static void Draw(SampleRenderer renderer)
        {
            ImGui.SetNextWindowPos(new Vector2(10, 10), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowBgAlpha(0.85f);

            ImGuiWindowFlags flags = ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.AlwaysAutoResize;
            if (!ImGui.Begin("Sample15", flags))
            {
                ImGui.End();
                return;
            }

            DrawSceneSection(renderer);
            ImGui.Separator();
            DrawRenderSettingsSection(renderer);
            ImGui.Separator();
            DrawOptionsSection(renderer);
            ImGui.Separator();
            DrawTonemapSection(renderer);
            ImGui.Separator();
            DrawStatsSection(renderer);
            ImGui.Separator();
            DrawControlsSection();

            ImGui.End();
        }

        static void DrawSceneSection(SampleRenderer renderer)
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "SCENE");
            ImGui.TextWrapped(renderer.CurrentSceneName);
            if (ImGui.Button("< Prev"))
            {
                renderer.PreviousScene();
                System.Console.WriteLine($"[Scene] {renderer.CurrentSceneName}");
            }
            ImGui.SameLine();
            if (ImGui.Button("Next >"))
            {
                renderer.NextScene();
                System.Console.WriteLine($"[Scene] {renderer.CurrentSceneName}");
            }
        }

        static void DrawRenderSettingsSection(SampleRenderer renderer)
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "RENDER SETTINGS");

            DrawBounceRow("Bounces", renderer, r => r.MaxBounces, (r, v) => r.MaxBounces = v);

            if (ImGui.Button(renderer.UseMergedTrianglesGas ? "Merged GAS: ON" : "Merged GAS: OFF"))
            {
                renderer.UseMergedTrianglesGas = !renderer.UseMergedTrianglesGas;
                renderer.RebuildCurrentScene();
                System.Console.WriteLine($"[MergedGAS] {renderer.UseMergedTrianglesGas}");
            }
        }

        // A getter/setter delegate pair, kept from when this drew three near-identical
        // per-material-kind bounce rows (docs/SAMPLE15_PLAN.md Milestone M3 collapsed
        // them into one unified row) - still convenient for the single row that's left.
        static void DrawBounceRow(string label, SampleRenderer renderer, System.Func<SampleRenderer, int> get, System.Action<SampleRenderer, int> set)
        {
            int value = get(renderer);
            ImGui.Text($"{label}:");
            ImGui.SameLine(120);
            if (ImGui.Button($"-##{label}"))
            {
                set(renderer, System.Math.Max(1, value - 1));
                renderer.setCamera(renderer.camera);
                System.Console.WriteLine($"[{label}Bounces] {get(renderer)}");
            }
            ImGui.SameLine();
            ImGui.Text(value.ToString());
            ImGui.SameLine();
            if (ImGui.Button($"+##{label}"))
            {
                set(renderer, System.Math.Min(8, value + 1));
                renderer.setCamera(renderer.camera);
                System.Console.WriteLine($"[{label}Bounces] {get(renderer)}");
            }
        }

        static void DrawOptionsSection(SampleRenderer renderer)
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "OPTIONS");

            bool denoiserOn = renderer.DenoiserOn;
            if (ImGui.Checkbox("Denoiser", ref denoiserOn))
            {
                renderer.DenoiserOn = denoiserOn;
                System.Console.WriteLine($"[Denoiser] {renderer.DenoiserOn}");
            }

            bool accumulate = renderer.Accumulate;
            if (ImGui.Checkbox("Accumulate", ref accumulate))
            {
                renderer.Accumulate = accumulate;
                System.Console.WriteLine($"[Accumulate] {renderer.Accumulate}");
            }

            // NEE/MIS on/off toggle (docs/SAMPLE15_PLAN.md Milestone M8) - for A/B
            // variance comparison against indirect-only lighting, per the plan's own
            // M4/M5 verification bar.
            bool neeEnabled = renderer.NeeEnabled;
            if (ImGui.Checkbox("NEE/MIS", ref neeEnabled))
            {
                renderer.NeeEnabled = neeEnabled;
                renderer.setCamera(renderer.camera);
                System.Console.WriteLine($"[NEE/MIS] {renderer.NeeEnabled}");
            }
        }

        // Tonemap + env-map controls (docs/SAMPLE15_PLAN.md Milestone M8). Env-map
        // rotation/intensity affect every scene's launch params regardless of whether
        // the active scene actually has one (SceneData.EnvMapPath unset) - harmless,
        // since LaunchParams.EnvMapWidth == 0 short-circuits both
        // EnvironmentMapSampling call sites in that case.
        static readonly string[] TonemapOperatorNames = { "Reinhard", "ACES" };

        static void DrawTonemapSection(SampleRenderer renderer)
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "TONEMAP");

            float exposure = renderer.ExposureStops;
            if (ImGui.SliderFloat("Exposure (stops)", ref exposure, -6f, 6f))
                renderer.ExposureStops = exposure;

            int op = renderer.TonemapOperator;
            if (ImGui.Combo("Operator", ref op, TonemapOperatorNames, TonemapOperatorNames.Length))
                renderer.TonemapOperator = op;

            float envRotationDeg = renderer.EnvMapRotation * (180f / System.MathF.PI);
            if (ImGui.SliderFloat("Env rotation (deg)", ref envRotationDeg, -180f, 180f))
                renderer.EnvMapRotation = envRotationDeg * (System.MathF.PI / 180f);

            float envIntensity = renderer.EnvMapIntensityMultiplier;
            if (ImGui.SliderFloat("Env intensity", ref envIntensity, 0f, 4f))
                renderer.EnvMapIntensityMultiplier = envIntensity;
        }

        static void DrawStatsSection(SampleRenderer renderer)
        {
            var stats = renderer.LastStats;
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "STATS");
            ImGui.Text($"{stats.Fps:0.0} fps ({stats.TotalFrameMs:0.00} ms)");
            ImGui.Text($"Samples accumulated: {stats.SamplesAccumulated}");
            ImGui.Text($"Trace: {stats.TraceMs:0.00}ms  Denoise: {stats.DenoiseMs:0.00}ms  Tonemap: {stats.TonemapMs:0.00}ms");
        }

        static void DrawControlsSection()
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "CONTROLS");
            ImGui.TextUnformatted("WASD move, hold Left Mouse to look");
            ImGui.TextUnformatted("[ / ] prev/next scene, M merged-GAS");
            ImGui.TextUnformatted("1/2 bounces -/+, Space denoiser, Tab accumulate");
            ImGui.TextUnformatted("N toggle NEE/MIS, T toggle tonemap operator");
            ImGui.TextUnformatted("Esc quit");
        }
    }
}
