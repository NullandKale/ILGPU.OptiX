using ImGuiNET;
using System.Numerics;

namespace Sample21.UI
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
            if (!ImGui.Begin("Sample21", flags))
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

            ImGui.Text($"Triangles: {renderer.TriangleCount}   Materials: {renderer.MaterialCount}");
            ImGui.Text($"Lights: {renderer.PointLightCount} point, {renderer.EmissiveTriangleLightCount} emissive tri" +
                (renderer.HasEnvMapLight ? ", 1 env map" : ""));
            ImGui.TextWrapped(renderer.AccelStructureSummary);
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

            ImGui.Spacing();
            DrawResolutionControls(renderer);
        }

        // Render resolution vs. window resolution (SampleRenderer.RenderScale/
        // AutoScaleEnabled/TargetFrameRate) - the manual slider and the auto-scaler
        // write to the same underlying RenderScale, so the slider always shows
        // whatever's actually in effect even while auto is driving it; it's just
        // non-interactive (BeginDisabled) while auto owns it, matching "auto off by
        // default, manual otherwise". Rays/sec is shown as a read-only measurement
        // only - it's not itself a target (see MeasuredRaysPerSecond's own comment on
        // why raw throughput doesn't make sense as one); the auto-scaler targets FPS.
        // "(?)" hover-tooltip marker - ImGui's standard convention for an inline help
        // hint that doesn't take up permanent panel space.
        static void HelpMarker(string text)
        {
            ImGui.SameLine();
            ImGui.TextDisabled("(?)");
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.PushTextWrapPos(ImGui.GetFontSize() * 25f);
                ImGui.TextUnformatted(text);
                ImGui.PopTextWrapPos();
                ImGui.EndTooltip();
            }
        }

        static void DrawResolutionControls(SampleRenderer renderer)
        {
            ImGui.Text($"Render res: {renderer.RenderWidth}x{renderer.RenderHeight}" +
                $" ({renderer.RenderScale * 100f:0}% of {renderer.WindowWidth}x{renderer.WindowHeight})");
            ImGui.Text($"Rays/sec: {renderer.MeasuredRaysPerSecond / 1_000_000.0:0.0}M");
            HelpMarker("Measured primary-ray throughput (render width x height x samples-per-pixel, divided by trace time) - a diagnostic readout only, not something you set or target directly.");

            bool autoScale = renderer.AutoScaleEnabled;
            if (ImGui.Checkbox("Auto (target FPS)", ref autoScale))
            {
                renderer.AutoScaleEnabled = autoScale;
                System.Console.WriteLine($"[AutoScale] {renderer.AutoScaleEnabled}");
            }
            HelpMarker("When on, automatically raises/lowers the render scale below to try to hit the Target FPS slider, checking every ~45 frames. When off, Render scale % is set manually instead.");

            ImGui.BeginDisabled(renderer.AutoScaleEnabled);
            float renderScalePercent = renderer.RenderScale * 100f;
            if (ImGui.SliderFloat("Render scale %", ref renderScalePercent, SampleRenderer.MinRenderScale * 100f, 100f, "%.0f%%"))
                renderer.RenderScale = renderScalePercent / 100f;
            ImGui.EndDisabled();
            HelpMarker("The traced/denoised image's own resolution as a percentage of the window's actual pixel size - lower values trade a softer (upscaled) image for a faster render. Independent of window size, so resizing the window doesn't change this percentage.");

            ImGui.BeginDisabled(!renderer.AutoScaleEnabled);
            float targetFrameRate = renderer.TargetFrameRate;
            if (ImGui.SliderFloat("Target FPS", ref targetFrameRate, 15f, 144f, "%.0f"))
                renderer.TargetFrameRate = targetFrameRate;
            ImGui.EndDisabled();
            HelpMarker("Only used when Auto is checked above - the frame rate the auto-scaler tries to sustain by adjusting Render scale %.");
        }

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

            // NRC - trains a neural radiance cache as a side channel off the existing
            // path tracer's own ground truth; does not affect the primary render (see
            // Device/CompositeProgram.cs's doc comment). On by default
            // (SampleRenderer.NrcEnabled's own default).
            bool nrcEnabled = renderer.NrcEnabled;
            if (ImGui.Checkbox("NRC (train cache)", ref nrcEnabled))
            {
                renderer.NrcEnabled = nrcEnabled;
                System.Console.WriteLine($"[NRC] {renderer.NrcEnabled}");
            }

            // Only meaningful while NRC is actually training - shows exactly what the
            // cache predicts instead of the primary render (see
            // SampleRenderer.ShowNrcPreview's own doc comment).
            bool showNrcPreview = renderer.ShowNrcPreview;
            if (ImGui.Checkbox("Show NRC cache preview", ref showNrcPreview))
            {
                renderer.ShowNrcPreview = showNrcPreview;
                System.Console.WriteLine($"[NRC Preview] {renderer.ShowNrcPreview}");
            }

            // Live network-tuning controls - lets settings be swept without a rebuild.
            // Defaults come from Core/NrcConstants.cs.
            ImGui.Indent();

            float lr = renderer.NrcLearningRate;
            if (ImGui.SliderFloat("Learning rate", ref lr, 0.00001f, 0.01f, "%.5f", ImGuiSliderFlags.Logarithmic))
                renderer.NrcLearningRate = lr;

            float weightDecay = renderer.NrcWeightDecay;
            if (ImGui.SliderFloat("Weight decay", ref weightDecay, 0f, 0.01f, "%.5f"))
                renderer.NrcWeightDecay = weightDecay;

            // VkNRC's "Use EMA" debug checkbox (SampleRenderer.NrcUseEmaInference's own
            // doc comment) - without it, the EMA alpha slider below adjusts a snapshot
            // nothing samples from.
            bool useEmaInference = renderer.NrcUseEmaInference;
            if (ImGui.Checkbox("Use EMA weights", ref useEmaInference))
            {
                renderer.NrcUseEmaInference = useEmaInference;
                System.Console.WriteLine($"[NRC EMA inference] {renderer.NrcUseEmaInference}");
            }

            float emaAlpha = renderer.NrcEmaAlpha;
            if (ImGui.SliderFloat("EMA alpha", ref emaAlpha, 0f, 0.999f, "%.3f"))
                renderer.NrcEmaAlpha = emaAlpha;

            float trainProb = renderer.NrcTrainProbability;
            if (ImGui.SliderFloat("Train probability", ref trainProb, 0f, 1f, "%.3f"))
                renderer.NrcTrainProbability = trainProb;

            // VkNRC's "C" adaptive-handoff footprint constant - RaygenProgram.cs decides
            // the handoff bounce adaptively per-path from this (see its own NRC
            // comment). Logarithmic since the useful range spans orders of magnitude
            // (smaller = hands off earlier/more aggressively).
            float footprintConstant = renderer.NrcFootprintConstant;
            if (ImGui.SliderFloat("Footprint constant", ref footprintConstant, 0.0001f, 1f, "%.4f", ImGuiSliderFlags.Logarithmic))
                renderer.NrcFootprintConstant = footprintConstant;

            // Performance knobs - both default to this sample's "lean" values, not
            // VkNRC's train-everything-every-frame originals.
            int trainBatches = renderer.NrcTrainBatchesPerFrame;
            if (ImGui.SliderInt("Train batches/frame", ref trainBatches, 1, Core.NrcConstants.TrainBatchesPerFrame))
                renderer.NrcTrainBatchesPerFrame = trainBatches;

            int trainEveryN = renderer.NrcTrainEveryNFrames;
            if (ImGui.SliderInt("Train every N frames", ref trainEveryN, 1, 8))
                renderer.NrcTrainEveryNFrames = trainEveryN;

            ImGui.Unindent();
        }

        // Tonemap + env-map controls. Env-map
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
            // Frames rendered since the current scene loaded - not a convergence
            // measure by itself, since accumulation history is tracked per-pixel
            // (LaunchParams.AccumCountBuffer) rather than as one global count.
            ImGui.Text($"Frames rendered: {stats.SamplesAccumulated}");
            ImGui.Text($"Trace: {stats.TraceMs:0.00}ms  Denoise: {stats.DenoiseMs:0.00}ms  Tonemap: {stats.TonemapMs:0.00}ms");
        }

        static void DrawControlsSection()
        {
            ImGui.TextColored(new Vector4(0.7f, 0.85f, 1f, 1f), "CONTROLS");
            ImGui.TextUnformatted("WASD move, hold Left Mouse to look");
            ImGui.TextUnformatted("[ / ] prev/next scene, M merged-GAS");
            ImGui.TextUnformatted("1/2 bounces -/+, Space denoiser, Tab accumulate");
            ImGui.TextUnformatted("R auto render-scale, T toggle tonemap operator");
            ImGui.TextUnformatted("Esc quit");
        }
    }
}
