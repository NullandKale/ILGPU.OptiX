using ImGuiNET;
using System.Numerics;

namespace Sample13.UI
{
    /// <summary>
    /// The ImGui control panel - four sections (SCENE, RENDER SETTINGS, OPTIONS,
    /// CONTROLS) plus a stats readout, laid out top-left over the rendered frame.
    /// Every control here has a keyboard equivalent too (see RenderWindow's
    /// UpdateOneShotKeys) - this panel is a visual/mouse-driven alternative, not a
    /// replacement for the keymap.
    /// </summary>
    public static class UiPanel
    {
        public static void Draw(SampleRenderer renderer)
        {
            ImGui.SetNextWindowPos(new Vector2(10, 10), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowBgAlpha(0.85f);

            ImGuiWindowFlags flags = ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.AlwaysAutoResize;
            if (!ImGui.Begin("Sample13", flags))
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

            DrawBounceRow("Mirror", renderer, r => r.MaxMirrorBounces, (r, v) => r.MaxMirrorBounces = v);
            DrawBounceRow("Refraction", renderer, r => r.MaxRefractionBounces, (r, v) => r.MaxRefractionBounces = v);
            DrawBounceRow("Diffuse", renderer, r => r.MaxDiffuseBounces, (r, v) => r.MaxDiffuseBounces = v);

            if (ImGui.Button(renderer.UseMergedTrianglesGas ? "Merged GAS: ON" : "Merged GAS: OFF"))
            {
                renderer.UseMergedTrianglesGas = !renderer.UseMergedTrianglesGas;
                renderer.RebuildCurrentScene();
                System.Console.WriteLine($"[MergedGAS] {renderer.UseMergedTrianglesGas}");
            }
        }

        // A getter/setter delegate pair keeps the three bounce rows from being three
        // near-identical copy-pasted blocks.
        static void DrawBounceRow(string label, SampleRenderer renderer, System.Func<SampleRenderer, int> get, System.Action<SampleRenderer, int> set)
        {
            int value = get(renderer);
            ImGui.Text($"{label}:");
            ImGui.SameLine(120);
            if (ImGui.Button($"-##{label}"))
            {
                set(renderer, System.Math.Max(0, value - 1));
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
            ImGui.TextUnformatted("1-6 bounce -/+, Space denoiser, Tab accumulate");
            ImGui.TextUnformatted("Esc quit");
        }
    }
}
