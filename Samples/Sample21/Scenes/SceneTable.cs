using System;

namespace Sample21
{
    /// <summary>
    /// The scene roster the renderer cycles through (I/U keys), mirroring the
    /// reference's RaytraceEntity.BuildSceneTable - lazily built, cached per index by
    /// the renderer.
    /// </summary>
    public static class SceneTable
    {
        public static Func<SceneData>[] Build() => new Func<SceneData>[]
        {
            BasicScenes.BuildDebugMaterialsScene,
            BasicScenes.BuildGgxRoughnessSweepScene,
            BasicScenes.BuildTextureTestScene,
            BasicScenes.BuildSimpleTestScene,
            BasicScenes.BuildNoAmbientLightDemoScene,
            ShowcaseScenes.BuildDemoScene,
            ShowcaseScenes.BuildCornellBox,
            ShowcaseScenes.BuildMirrorSpheresOnChecker,
            ShowcaseScenes.BuildCylindersDisksAndTriangles,
            ShowcaseScenes.BuildBoxesShowcase,
            MeshScenes.BuildAllMeshesScene,
            MeshScenes.BuildBunnyScene,
            MeshScenes.BuildTeapotScene,
            MeshScenes.BuildCowScene,
            MeshScenes.BuildDragonScene,
            MeshScenes.BuildSponzaScene,
            PbrShowcaseScene.BuildPbrShowcaseScene,
            // MuseumScene/RadialMuseumScene are video-textured and deferred - they
            // need Media/VideoReader.cs + VideoTexture.cs + OpenCvSharp + ffmpeg.exe,
            // none of which are ported yet.
        };
    }
}
