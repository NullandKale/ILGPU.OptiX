using ILGPU.OptiX;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Sample13
{
    /// <summary>
    /// Per-scene texture ownership: loads image textures into CudaTextureObjects and
    /// video paths into per-frame-refreshed VideoTextures, keyed by relative path so a
    /// texture referenced by multiple materials is only loaded once. Cleared (all
    /// GPU objects disposed) on every scene switch - only the active scene's textures
    /// stay resident.
    /// </summary>
    public sealed class TextureCache
    {
        static readonly string[] VideoExtensions = { ".mp4", ".avi", ".mov", ".mkv" };

        readonly Dictionary<string, ulong> textureCache = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase);
        readonly List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        readonly Dictionary<string, VideoTexture> videoTextureCache = new Dictionary<string, VideoTexture>(StringComparer.OrdinalIgnoreCase);
        readonly List<VideoTexture> activeVideoTextures = new List<VideoTexture>();

        public bool HasActiveVideos => activeVideoTextures.Count > 0;

        // Loads (or reuses an already-loaded) texture for a relative path under the
        // output directory - same TextureLoader.LoadRgba8/CudaTextureObject pattern as
        // Sample08. Paths with a known video extension are routed to a VideoTexture
        // instead. A missing file degrades to handle 0 ("no texture") rather than
        // crashing the scene switch.
        public ulong GetOrLoad(string relativePath)
        {
            if (VideoExtensions.Contains(Path.GetExtension(relativePath), StringComparer.OrdinalIgnoreCase))
                return GetOrLoadVideo(relativePath);

            if (textureCache.TryGetValue(relativePath, out var cachedHandle))
                return cachedHandle;

            string fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (!File.Exists(fullPath))
            {
                Console.WriteLine($"[Warning] Texture not found, material will use its flat diffuse color instead: {fullPath}");
                textureCache[relativePath] = 0;
                return 0;
            }

            var pixels = TextureLoader.LoadRgba8(fullPath, out var texWidth, out var texHeight);
            LogAlphaStats(relativePath, pixels, texWidth, texHeight);
            var textureObject = new CudaTextureObject(pixels, texWidth, texHeight);
            textureObjects.Add(textureObject);
            var handle = textureObject.TextureObject;
            textureCache[relativePath] = handle;
            return handle;
        }

        // Diagnostic for alpha-cutout materials (e.g. Sponza's leaf texture) - reports
        // whether the loaded pixel data actually has varying alpha at all (a fully-255
        // texture means whatever alpha-cutout look a material expects can't come from
        // this file, regardless of how the shading code samples it).
        static void LogAlphaStats(string relativePath, byte[] rgba, int width, int height)
        {
            byte minA = 255, maxA = 0;
            long transparentCount = 0;
            for (int i = 3; i < rgba.Length; i += 4)
            {
                byte a = rgba[i];
                if (a < minA) minA = a;
                if (a > maxA) maxA = a;
                if (a < 128) transparentCount++;
            }
            long pixelCount = (long)width * height;
            Console.WriteLine($"[TextureCache] {relativePath}: {width}x{height} alpha[min={minA} max={maxA}] {transparentCount}/{pixelCount} pixels <128 alpha ({(100.0 * transparentCount / pixelCount):F1}%)");
        }

        // Refreshes every active video texture with its latest decoded frame - called
        // once per rendered frame by the animation step.
        public void RefreshVideos()
        {
            foreach (var videoTexture in activeVideoTextures)
                videoTexture.Refresh();
        }

        public void Clear()
        {
            foreach (var textureObject in textureObjects)
                textureObject.Dispose();
            textureObjects.Clear();
            textureCache.Clear();

            foreach (var videoTexture in videoTextureCache.Values)
                videoTexture.Dispose();
            videoTextureCache.Clear();
            activeVideoTextures.Clear();
        }

        // Loads (or reuses) a VideoTexture - same cache-by-path convention as images,
        // but tracked separately in activeVideoTextures so RefreshVideos knows which
        // textures need a per-frame update. Requires ffmpeg.exe on PATH; if the video
        // file itself is missing or unreadable, degrades the same way a missing image
        // does (TextureObject stays 0 - untextured material).
        ulong GetOrLoadVideo(string relativePath)
        {
            if (videoTextureCache.TryGetValue(relativePath, out var cached))
            {
                if (!activeVideoTextures.Contains(cached))
                    activeVideoTextures.Add(cached);
                return cached.TextureObject;
            }

            string fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (!File.Exists(fullPath))
            {
                Console.WriteLine($"[Warning] Video texture not found, material will use its flat diffuse color instead: {fullPath}");
                return 0;
            }

            VideoTexture videoTexture;
            try
            {
                videoTexture = new VideoTexture(fullPath);
            }
            catch (Exception ex)
            {
                // ffmpeg.exe missing from PATH, or OpenCvSharp couldn't probe the file -
                // degrade to "no texture" rather than crashing the whole scene switch.
                Console.Error.WriteLine($"[VideoTexture] Failed to open '{fullPath}': {ex.Message}");
                return 0;
            }

            videoTextureCache[relativePath] = videoTexture;
            activeVideoTextures.Add(videoTexture);
            return videoTexture.TextureObject;
        }
    }
}
