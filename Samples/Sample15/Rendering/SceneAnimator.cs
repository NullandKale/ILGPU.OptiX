using ILGPU.Algorithms;
using System;
using System.Diagnostics;

namespace Sample15
{
    /// <summary>
    /// Per-frame animation for the museum/radial-museum scenes - direct port of the
    /// reference's OrbitingLightEntity/PulsingLightEntity/BobbingSphereEntity update
    /// formulas (RayTracing/Scenes/TestScenesRandom.cs), evaluated from elapsed
    /// wall-clock time rather than accumulated per-frame dt: each entity's own
    /// "t += dt" accumulator, started at 0 and monotonically advanced once per Update
    /// call, is mathematically just elapsed time since the scene started, so this is
    /// equivalent while being immune to any frame-timing drift.
    ///
    /// The two host-side arrays are private per-frame working copies, refreshed at
    /// every scene switch - never the same array instances as the cached SceneData,
    /// since that same SceneData is reused verbatim if the user cycles back to the
    /// scene later.
    /// </summary>
    public sealed class SceneAnimator
    {
        readonly SceneGpuBuffers buffers;
        readonly AccelStructureBuilder accel;
        readonly TextureCache textures;

        PointLightGpu[] animatedLightsHost = Array.Empty<PointLightGpu>();
        SphereData[] animatedSpheresHost = Array.Empty<SphereData>();
        readonly Stopwatch animationClock = new Stopwatch();

        public SceneAnimator(SceneGpuBuffers buffers, AccelStructureBuilder accel, TextureCache textures)
        {
            this.buffers = buffers;
            this.accel = accel;
            this.textures = textures;
        }

        public void OnSceneSwitched(SceneData scene)
        {
            animatedLightsHost = (PointLightGpu[])scene.Lights.Clone();
            animatedSpheresHost = (SphereData[])scene.Spheres.Clone();
            animationClock.Restart();
        }

        // Sample13's version also OR's in textures.HasActiveVideos (video textures
        // don't move geometry but still need a per-frame refresh/accumulation reset) -
        // not applicable here since video-texture scenes are deferred (see
        // docs/SAMPLE14_PLAN.md) and TextureCache has no video support in Sample14.
        public bool IsAnimated(SceneData scene) => scene.HasAnyAnimation;

        // Called every rendered frame under the renderer's gpuLock, before OptixLaunch.
        public void Update(SceneData scene)
        {
            float t = (float)animationClock.Elapsed.TotalSeconds;
            bool lightsChanged = false;

            foreach (var anim in scene.PulsingLights)
            {
                float s = 0.5f + (0.5f * XMath.Sin(anim.Speed * t));
                float mult = Math.Max(0f, anim.MinMult + ((anim.MaxMult - anim.MinMult) * s));
                var light = animatedLightsHost[anim.LightIndex];
                light.Intensity = anim.BaseIntensity * mult;
                animatedLightsHost[anim.LightIndex] = light;
                lightsChanged = true;
            }

            foreach (var anim in scene.OrbitingLights)
            {
                float a = (anim.AngularSpeed * t) + anim.Phase;
                var light = animatedLightsHost[anim.LightIndex];
                light.Position = new Vec3(
                    anim.Pivot.x + (anim.Radius * XMath.Cos(a)),
                    anim.Height,
                    anim.Pivot.z + (anim.Radius * XMath.Sin(a)));
                animatedLightsHost[anim.LightIndex] = light;
                lightsChanged = true;
            }

            if (lightsChanged)
                buffers.UpdateLights(animatedLightsHost);

            if (scene.BobbingSpheres.Length > 0)
            {
                foreach (var anim in scene.BobbingSpheres)
                {
                    float y = anim.BaseY + (anim.Amplitude * XMath.Sin((anim.Speed * t) + anim.Phase));
                    var sphere = animatedSpheresHost[anim.SphereIndex];
                    sphere.Center = new Vec3(sphere.Center.x, y, sphere.Center.z);
                    animatedSpheresHost[anim.SphereIndex] = sphere;
                }

                buffers.UpdateAnimatedSpheres(animatedSpheresHost);
                accel.RefitCustomPrimitives(scene);
            }
        }
    }
}
