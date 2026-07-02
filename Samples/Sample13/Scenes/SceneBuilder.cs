using System;
using System.Collections.Generic;
using System.IO;

namespace Sample13
{
    /// <summary>
    /// Accumulates geometry, materials, lights, and animation descriptors for one scene
    /// and produces the final <see cref="SceneData"/> via <see cref="Build"/>. Replaces
    /// the per-scene copies of AddQuad/AddTriangle/AddMaterial/AddMesh/... local
    /// functions the scene builders used to carry around individually.
    /// </summary>
    public sealed class SceneBuilder
    {
        // Scene-level settings - defaults match SceneData's own field defaults, so a
        // scene only needs to set what it cares about.
        public string Name = "";
        public Vec3 AmbientColor = new Vec3(1f, 1f, 1f);
        public float AmbientIntensity = 0.05f;
        public Vec3 BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f);
        public Vec3 BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f);
        public Vec3 CameraOrigin;
        public Vec3 CameraLookAt;
        public Vec3 CameraUp = new Vec3(0f, 1f, 0f);
        public float CameraFovDeg = 50f;
        public float CameraWorldScale = 10f;

        readonly List<Vec3> vertices = new List<Vec3>();
        readonly List<Vec3> normals = new List<Vec3>();
        readonly List<Vec2> texCoords = new List<Vec2>();
        readonly List<Vec3i> indices = new List<Vec3i>();
        readonly List<uint> triangleMaterialIds = new List<uint>();
        readonly List<MeshRange> meshRanges = new List<MeshRange>();

        readonly List<MaterialSbtData> materials = new List<MaterialSbtData>();
        readonly List<string> materialTexturePaths = new List<string>();

        readonly List<SphereData> spheres = new List<SphereData>();
        readonly List<uint> sphereMaterialIds = new List<uint>();
        readonly List<BoxData> boxes = new List<BoxData>();
        readonly List<uint> boxMaterialIds = new List<uint>();
        readonly List<CylinderYData> cylindersY = new List<CylinderYData>();
        readonly List<uint> cylinderYMaterialIds = new List<uint>();
        readonly List<DiskData> disks = new List<DiskData>();
        readonly List<uint> diskMaterialIds = new List<uint>();
        readonly List<RectData> xyRects = new List<RectData>();
        readonly List<uint> xyRectMaterialIds = new List<uint>();
        readonly List<RectData> xzRects = new List<RectData>();
        readonly List<uint> xzRectMaterialIds = new List<uint>();
        readonly List<RectData> yzRects = new List<RectData>();
        readonly List<uint> yzRectMaterialIds = new List<uint>();

        readonly List<PointLightGpu> lights = new List<PointLightGpu>();
        readonly List<OrbitingLightAnim> orbitingLights = new List<OrbitingLightAnim>();
        readonly List<PulsingLightAnim> pulsingLights = new List<PulsingLightAnim>();
        readonly List<BobbingSphereAnim> bobbingSpheres = new List<BobbingSphereAnim>();

        uint[] voxelMaterialIds = Array.Empty<uint>();
        Vec3 volumeGridMin;
        Vec3 volumeVoxelSize = new Vec3(1f, 1f, 1f);
        Vec3i volumeDims;

        // Read access for scenes that need to inspect what they've placed so far
        // (BuildDemoScene's sphere overlap rejection sampling).
        public IReadOnlyList<SphereData> Spheres => spheres;

        public int AddMaterial(MaterialSbtData material, string texturePath = null)
        {
            materials.Add(material);
            materialTexturePaths.Add(texturePath);
            return materials.Count - 1;
        }

        public void AddQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, uint materialId)
        {
            int baseIndex = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c); vertices.Add(d);
            normals.Add(normal); normals.Add(normal); normals.Add(normal); normals.Add(normal);
            texCoords.Add(new Vec2(0, 0)); texCoords.Add(new Vec2(1, 0));
            texCoords.Add(new Vec2(1, 1)); texCoords.Add(new Vec2(0, 1));

            indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
            triangleMaterialIds.Add(materialId);
            indices.Add(new Vec3i(baseIndex, baseIndex + 2, baseIndex + 3));
            triangleMaterialIds.Add(materialId);
        }

        public void AddTriangle(Vec3 a, Vec3 b, Vec3 c, Vec3 normal, uint materialId)
        {
            int baseIndex = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c);
            normals.Add(normal); normals.Add(normal); normals.Add(normal);
            texCoords.Add(new Vec2(0f, 0f)); texCoords.Add(new Vec2(1f, 0f)); texCoords.Add(new Vec2(0f, 1f));

            indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
            triangleMaterialIds.Add(materialId);
        }

        // Face normal computed from winding, like the radial museum's own AddTriangle.
        public void AddTriangle(Vec3 a, Vec3 b, Vec3 c, uint materialId) =>
            AddTriangle(a, b, c, Vec3.unitVector(Vec3.cross(b - a, c - a)), materialId);

        public int AddSphere(Vec3 center, float radius, uint materialId)
        {
            spheres.Add(new SphereData { Center = center, Radius = radius });
            sphereMaterialIds.Add(materialId);
            return spheres.Count - 1;
        }

        public void AddBox(Vec3 min, Vec3 max, uint materialId)
        {
            boxes.Add(new BoxData { Min = min, Max = max });
            boxMaterialIds.Add(materialId);
        }

        public void AddCylinderY(Vec3 center, float radius, float yMin, float yMax, uint materialId, bool capped = true)
        {
            cylindersY.Add(new CylinderYData { Center = center, Radius = radius, YMin = yMin, YMax = yMax, Capped = capped ? 1 : 0 });
            cylinderYMaterialIds.Add(materialId);
        }

        public void AddDisk(Vec3 center, Vec3 normal, float radius, uint materialId)
        {
            disks.Add(new DiskData { Center = center, Normal = normal, Radius = radius });
            diskMaterialIds.Add(materialId);
        }

        public void AddXYRect(float a0, float a1, float b0, float b1, float c, uint materialId)
        {
            xyRects.Add(new RectData { A0 = a0, A1 = a1, B0 = b0, B1 = b1, C = c });
            xyRectMaterialIds.Add(materialId);
        }

        public void AddXZRect(float a0, float a1, float b0, float b1, float c, uint materialId)
        {
            xzRects.Add(new RectData { A0 = a0, A1 = a1, B0 = b0, B1 = b1, C = c });
            xzRectMaterialIds.Add(materialId);
        }

        public void AddYZRect(float a0, float a1, float b0, float b1, float c, uint materialId)
        {
            yzRects.Add(new RectData { A0 = a0, A1 = a1, B0 = b0, B1 = b1, C = c });
            yzRectMaterialIds.Add(materialId);
        }

        public int AddLight(Vec3 position, Vec3 color, float intensity)
        {
            lights.Add(new PointLightGpu { Position = position, Color = color, Intensity = intensity });
            return lights.Count - 1;
        }

        public void AddOrbitingLight(int lightIndex, Vec3 pivot, float radius, float height, float angularSpeed, float phase) =>
            orbitingLights.Add(new OrbitingLightAnim { LightIndex = lightIndex, Pivot = pivot, Radius = radius, Height = height, AngularSpeed = angularSpeed, Phase = phase });

        public void AddPulsingLight(int lightIndex, float ampFraction, float speed) =>
            pulsingLights.Add(new PulsingLightAnim { LightIndex = lightIndex, BaseIntensity = lights[lightIndex].Intensity, MinMult = Math.Max(0f, 1f - ampFraction), MaxMult = 1f + ampFraction, Speed = speed });

        public void AddBobbingSphere(int sphereIndex, float amplitude, float speed, float phase) =>
            bobbingSpheres.Add(new BobbingSphereAnim { SphereIndex = sphereIndex, BaseY = spheres[sphereIndex].Center.y, Amplitude = amplitude, Speed = speed, Phase = phase });

        public void SetVolumeGrid(uint[] voxelIds, Vec3 gridMin, Vec3 voxelSize, Vec3i dims)
        {
            voxelMaterialIds = voxelIds;
            volumeGridMin = gridMin;
            volumeVoxelSize = voxelSize;
            volumeDims = dims;
        }

        public static void ComputeBounds(Vec3[] meshVertices, out Vec3 min, out Vec3 max)
        {
            min = new Vec3(float.MaxValue, float.MaxValue, float.MaxValue);
            max = new Vec3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in meshVertices)
            {
                min = new Vec3(Math.Min(v.x, min.x), Math.Min(v.y, min.y), Math.Min(v.z, min.z));
                max = new Vec3(Math.Max(v.x, max.x), Math.Max(v.y, max.y), Math.Max(v.z, max.z));
            }
        }

        // The "all meshes" placement variant: recenters the mesh on its bounding-box
        // center, uniformly scales its largest extent to targetSize, then drops it so
        // its lowest vertex sits at targetPos.y.
        public void AddMesh(string objFileName, Vec3 targetPos, uint materialId, float targetSize = 1.6f)
        {
            int indexStart = indices.Count;
            string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
            var mesh = OBJModel.Load(objPath);

            ComputeBounds(mesh.Vertices, out Vec3 min, out Vec3 max);
            Vec3 center = (min + max) / 2f;
            Vec3 size = max - min;
            float maxExtent = Math.Max(size.x, Math.Max(size.y, size.z));
            float scale = maxExtent > 1e-6f ? targetSize / maxExtent : 1f;

            int baseIndex = vertices.Count;
            float minY = float.MaxValue;
            var placed = new Vec3[mesh.Vertices.Length];
            for (var i = 0; i < mesh.Vertices.Length; i++)
            {
                var v = (mesh.Vertices[i] - center) * scale;
                placed[i] = v;
                minY = Math.Min(minY, v.y);
            }
            float groundOffset = targetPos.y - minY;
            for (var i = 0; i < placed.Length; i++)
            {
                vertices.Add(new Vec3(
                    placed[i].x + targetPos.x,
                    placed[i].y + groundOffset,
                    placed[i].z + targetPos.z));
            }
            normals.AddRange(mesh.Normals);
            texCoords.AddRange(mesh.TexCoords);

            foreach (var tri in mesh.Indices)
            {
                indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                triangleMaterialIds.Add(materialId);
            }
            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
        }

        // The museum placement variant: raw uniform scale (no normalization), dropped
        // so the lowest vertex sits at targetPos.y; silently skips a missing OBJ file
        // so museum scenes degrade gracefully instead of crashing the scene switch.
        public void AddMeshAutoGround(string objFileName, uint materialId, float scale, Vec3 targetPos)
        {
            string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
            if (!File.Exists(objPath))
                return;
            var mesh = OBJModel.Load(objPath);

            float minY = float.MaxValue;
            foreach (var v in mesh.Vertices)
                minY = Math.Min(minY, v.y * scale);
            float groundOffset = targetPos.y - minY;

            int indexStart = indices.Count;
            int baseIndex = vertices.Count;
            foreach (var v in mesh.Vertices)
                vertices.Add(new Vec3((v.x * scale) + targetPos.x, (v.y * scale) + groundOffset, (v.z * scale) + targetPos.z));
            normals.AddRange(mesh.Normals);
            texCoords.AddRange(mesh.TexCoords);
            foreach (var tri in mesh.Indices)
            {
                indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                triangleMaterialIds.Add(materialId);
            }
            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
        }

        // Fills any gaps between (and before/after) explicitly tracked mesh ranges
        // with anonymous ranges covering untracked indices (e.g. ad-hoc AddTriangle
        // calls interleaved between AddMeshAutoGround calls in the radial museum)
        // - guarantees the result partitions [0, totalIndexCount) with no gaps or
        // overlaps, which the triangles-GAS builder relies on (every triangle must
        // land in exactly one GAS build input).
        static MeshRange[] FillMeshRangeGaps(List<MeshRange> tracked, int totalIndexCount)
        {
            tracked.Sort((a, b) => a.IndexStart.CompareTo(b.IndexStart));
            var result = new List<MeshRange>();
            int cursor = 0;
            foreach (var range in tracked)
            {
                if (range.IndexStart > cursor)
                    result.Add(new MeshRange { IndexStart = cursor, IndexCount = range.IndexStart - cursor });
                result.Add(range);
                cursor = range.IndexStart + range.IndexCount;
            }
            if (cursor < totalIndexCount)
                result.Add(new MeshRange { IndexStart = cursor, IndexCount = totalIndexCount - cursor });
            return result.ToArray();
        }

        public SceneData Build() => new SceneData
        {
            Name = Name,
            Vertices = vertices.ToArray(),
            Normals = normals.ToArray(),
            TexCoords = texCoords.ToArray(),
            Indices = indices.ToArray(),
            TriangleMaterialIds = triangleMaterialIds.ToArray(),
            MeshRanges = meshRanges.Count > 0 ? FillMeshRangeGaps(meshRanges, indices.Count) : Array.Empty<MeshRange>(),
            Materials = materials.ToArray(),
            MaterialTexturePaths = materialTexturePaths.ToArray(),
            Spheres = spheres.ToArray(),
            SphereMaterialIds = sphereMaterialIds.ToArray(),
            Boxes = boxes.ToArray(),
            BoxMaterialIds = boxMaterialIds.ToArray(),
            CylindersY = cylindersY.ToArray(),
            CylinderYMaterialIds = cylinderYMaterialIds.ToArray(),
            Disks = disks.ToArray(),
            DiskMaterialIds = diskMaterialIds.ToArray(),
            XYRects = xyRects.ToArray(),
            XYRectMaterialIds = xyRectMaterialIds.ToArray(),
            XZRects = xzRects.ToArray(),
            XZRectMaterialIds = xzRectMaterialIds.ToArray(),
            YZRects = yzRects.ToArray(),
            YZRectMaterialIds = yzRectMaterialIds.ToArray(),
            VoxelMaterialIds = voxelMaterialIds,
            VolumeGridMin = volumeGridMin,
            VolumeVoxelSize = volumeVoxelSize,
            VolumeDims = volumeDims,
            Lights = lights.ToArray(),
            OrbitingLights = orbitingLights.ToArray(),
            PulsingLights = pulsingLights.ToArray(),
            BobbingSpheres = bobbingSpheres.ToArray(),
            AmbientColor = AmbientColor,
            AmbientIntensity = AmbientIntensity,
            BackgroundTop = BackgroundTop,
            BackgroundBottom = BackgroundBottom,
            CameraOrigin = CameraOrigin,
            CameraLookAt = CameraLookAt,
            CameraUp = CameraUp,
            CameraFovDeg = CameraFovDeg,
            CameraWorldScale = CameraWorldScale,
        };
    }
}
