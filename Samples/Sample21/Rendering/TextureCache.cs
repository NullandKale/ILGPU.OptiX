using ILGPU.OptiX;
using ILGPU.OptiX.Cuda;
using System;
using System.Collections.Generic;
using System.IO;

namespace Sample21
{
    /// <summary>
    /// Per-scene texture ownership: loads image textures into CudaTextureObjects, keyed
    /// by relative path so a texture referenced by multiple materials is only loaded
    /// once. Cleared (all GPU objects disposed) on every scene switch - only the active
    /// scene's textures stay resident. Only static images are loaded; no video-texture
    /// support.
    /// </summary>
    public sealed class TextureCache
    {
        readonly Dictionary<string, ulong> textureCache = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase);
        readonly List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        // Loads (or reuses an already-loaded) texture for a relative path under the
        // output directory. A missing file degrades to handle 0 ("no texture") rather
        // than crashing the scene switch.
        public ulong GetOrLoad(string relativePath)
        {
            if (textureCache.TryGetValue(relativePath, out var cachedHandle))
                return cachedHandle;

            string fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (!File.Exists(fullPath))
            {
                textureCache[relativePath] = 0;
                return 0;
            }

            var pixels = TextureLoader.LoadRgba8(fullPath, out var texWidth, out var texHeight);
            var textureObject = new CudaTextureObject(pixels, texWidth, texHeight);
            textureObjects.Add(textureObject);
            var handle = textureObject.TextureObject;
            textureCache[relativePath] = handle;
            return handle;
        }

        public void Clear()
        {
            foreach (var textureObject in textureObjects)
                textureObject.Dispose();
            textureObjects.Clear();
            textureCache.Clear();
        }
    }
}
