using System;

namespace Sample13
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
            BasicScenes.BuildDebugOrenNayarScene,
            BasicScenes.BuildTextureTestScene,
            BasicScenes.BuildSimpleTestScene,
            ShowcaseScenes.BuildDemoScene,
            ShowcaseScenes.BuildCornellBox,
            ShowcaseScenes.BuildMirrorSpheresOnChecker,
            ShowcaseScenes.BuildCylindersDisksAndTriangles,
            ShowcaseScenes.BuildBoxesShowcase,
            VolumeScenes.BuildVolumeGridTestScene,
            MeshScenes.BuildAllMeshesScene,
            MeshScenes.BuildBunnyScene,
            MeshScenes.BuildTeapotScene,
            MeshScenes.BuildCowScene,
            MeshScenes.BuildDragonScene,
            MeshScenes.BuildSponzaScene,
            // MuseumScene/RadialMuseumScene are video-textured and not implemented here
            // - they need Media/VideoReader.cs + VideoTexture.cs + OpenCvSharp +
            // ffmpeg.exe, none of which this sample has.
        };
    }
}
