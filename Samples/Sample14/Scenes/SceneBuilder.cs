using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;
using System.Collections.Generic;
using System.IO;
using ILGPU.OptiX.Cuda;

namespace Sample14
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
        public float AmbientIntensity = 0.1f;
        public Vec3 BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f);
        public Vec3 BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f);
        public string EnvMapPath;
        public float EnvMapIntensity = 1f;
        public Vec3 CameraOrigin;
        public Vec3 CameraLookAt;
        public Vec3 CameraUp = new Vec3(0f, 1f, 0f);
        public float CameraFovDeg = 50f;
        public float CameraWorldScale = 10f;

        readonly List<Vec3> vertices = new List<Vec3>();
        readonly List<Vec3> normals = new List<Vec3>();
        readonly List<Vec2> texCoords = new List<Vec2>();
        readonly List<Vec3> tangents = new List<Vec3>();
        readonly List<Vec3i> indices = new List<Vec3i>();
        readonly List<uint> triangleMaterialIds = new List<uint>();
        readonly List<MeshRange> meshRanges = new List<MeshRange>();

        readonly List<MaterialSbtData> materials = new List<MaterialSbtData>();
        readonly List<string> materialTexturePaths = new List<string>();
        readonly List<string> materialNormalTexturePaths = new List<string>();
        readonly List<string> materialOrmTexturePaths = new List<string>();

        readonly List<PointLightGpu> lights = new List<PointLightGpu>();
        readonly List<OrbitingLightAnim> orbitingLights = new List<OrbitingLightAnim>();
        readonly List<PulsingLightAnim> pulsingLights = new List<PulsingLightAnim>();

        // Placed sphere-mesh centers/radii, tracked purely for overlap rejection
        // sampling (BuildDemoScene) - not a GAS-facing type, just bookkeeping.
        readonly List<(Vec3 Center, float Radius)> sphereMeshes = new List<(Vec3, float)>();
        public IReadOnlyList<(Vec3 Center, float Radius)> Spheres => sphereMeshes;

        public int AddMaterial(MaterialSbtData material, string texturePath = null, string normalTexturePath = null, string ormTexturePath = null)
        {
            materials.Add(material);
            materialTexturePaths.Add(texturePath);
            materialNormalTexturePaths.Add(normalTexturePath);
            materialOrmTexturePaths.Add(ormTexturePath);
            return materials.Count - 1;
        }

        // V is flipped relative to the geometric a/b vs. c/d edges (a/b get V=1, c/d
        // get V=0) to match every other UV source in this sample: CudaTextureObject/
        // TextureLoader upload pixel row 0 (the image's top row) first, and CUDA's
        // tex2D samples V=0 from that same first row - so V=0 must mean "top of the
        // image", exactly the reason Model.cs's own OBJ loader flips "vt" lines (see
        // its "vt" case comment).
        public void AddQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, uint materialId)
        {
            int baseIndex = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c); vertices.Add(d);
            normals.Add(normal); normals.Add(normal); normals.Add(normal); normals.Add(normal);
            Vec2 uvA = new Vec2(0f, 1f), uvB = new Vec2(1f, 1f), uvC = new Vec2(1f, 0f), uvD = new Vec2(0f, 0f);
            texCoords.Add(uvA); texCoords.Add(uvB); texCoords.Add(uvC); texCoords.Add(uvD);
            // A planar quad has one constant tangent across all 4 corners - computed
            // from the same (a,b,c) triangle used for its first half.
            Vec3 tangent = TangentFromTriangle(a, b, c, uvA, uvB, uvC, normal);
            tangents.Add(tangent); tangents.Add(tangent); tangents.Add(tangent); tangents.Add(tangent);

            indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
            triangleMaterialIds.Add(materialId);
            indices.Add(new Vec3i(baseIndex, baseIndex + 2, baseIndex + 3));
            triangleMaterialIds.Add(materialId);
        }

        // Same V-flip as AddQuad, for the same reason.
        public void AddTriangle(Vec3 a, Vec3 b, Vec3 c, Vec3 normal, uint materialId)
        {
            int baseIndex = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c);
            normals.Add(normal); normals.Add(normal); normals.Add(normal);
            Vec2 uvA = new Vec2(0f, 1f), uvB = new Vec2(1f, 1f), uvC = new Vec2(0f, 0f);
            texCoords.Add(uvA); texCoords.Add(uvB); texCoords.Add(uvC);
            Vec3 tangent = TangentFromTriangle(a, b, c, uvA, uvB, uvC, normal);
            tangents.Add(tangent); tangents.Add(tangent); tangents.Add(tangent);

            indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
            triangleMaterialIds.Add(materialId);
        }

        // Direct-geometry counterpart to Model.cs's ComputeTangents - AddQuad/
        // AddTriangle already know their own UVs and normal up front, so a real
        // per-call tangent solve (not an arbitrary fallback) is just as cheap here.
        static Vec3 TangentFromTriangle(Vec3 a, Vec3 b, Vec3 c, Vec2 uvA, Vec2 uvB, Vec2 uvC, Vec3 normal)
        {
            Vec3 e1 = b - a, e2 = c - a;
            float du1 = uvB.x - uvA.x, dv1 = uvB.y - uvA.y;
            float du2 = uvC.x - uvA.x, dv2 = uvC.y - uvA.y;

            float denom = (du1 * dv2) - (du2 * dv1);
            Vec3 tangent = Math.Abs(denom) > 1e-12f
                ? (e1 * dv2) - (e2 * dv1)
                : (Math.Abs(normal.y) < 0.999f ? Vec3.cross(new Vec3(0f, 1f, 0f), normal) : Vec3.cross(new Vec3(1f, 0f, 0f), normal));

            tangent -= normal * Vec3.dot(normal, tangent);
            return tangent.lengthSquared() > 1e-12f ? Vec3.unitVector(tangent) : Vec3.unitVector(Vec3.cross(new Vec3(0f, 1f, 0f), normal));
        }

        // Face normal computed from winding, like the radial museum's own AddTriangle.
        public void AddTriangle(Vec3 a, Vec3 b, Vec3 c, uint materialId) =>
            AddTriangle(a, b, c, Vec3.unitVector(Vec3.cross(b - a, c - a)), materialId);

        // Procedural mesh generators - replace the old analytic custom-primitive GAS
        // path (spheres/boxes/cylinders/disks/rects) with real triangle geometry, so
        // every scene renders through the single triangles GAS (faster: no second GAS,
        // no IAS indirection, no per-ray hitKind dispatch in the closest-hit hot path).

        // UV sphere - one seam-duplicated vertex column at j==segments so each ring's
        // last quad still gets correct UVs, matching the classic songho.ca layout.
        // Winding is verified to face outward under this file's cross(b-a,c-a)
        // convention (see AddTriangle's face-normal comment).
        // Default 128x128 (~32.5k triangles) is intentionally high - a deliberate
        // choice for visual quality on hero/showcase spheres (mirror/glass
        // silhouettes are a true geometric edge, not smoothed by per-vertex normals,
        // so a low-poly sphere shows a visibly faceted outline up close). Pass a much
        // lower rings/segments explicitly for small/distant/background-clutter spheres
        // where that fidelity buys nothing (see ShowcaseScenes.BuildDemoScene's
        // 100-sphere loop) rather than lowering this default globally.
        public void AddSphereMesh(Vec3 center, float radius, uint materialId, int rings = 128, int segments = 128)
        {
            sphereMeshes.Add((center, radius));

            int indexStart = indices.Count;
            int baseIndex = vertices.Count;

            for (int i = 0; i <= rings; i++)
            {
                float theta = i / (float)rings * (float)Math.PI;
                float sinTheta = (float)Math.Sin(theta);
                float cosTheta = (float)Math.Cos(theta);
                for (int j = 0; j <= segments; j++)
                {
                    float phi = j / (float)segments * 2f * (float)Math.PI;
                    float sinPhi = (float)Math.Sin(phi);
                    float cosPhi = (float)Math.Cos(phi);

                    Vec3 dir = new Vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
                    vertices.Add(center + (radius * dir));
                    normals.Add(dir);
                    texCoords.Add(new Vec2(j / (float)segments, i / (float)rings));
                    Vec3 tangent = sinTheta > 1e-6f
                        ? new Vec3(-sinPhi, 0f, cosPhi)
                        : Vec3.unitVector(Vec3.cross(new Vec3(0f, 1f, 0f), dir));
                    tangents.Add(tangent);
                }
            }

            for (int i = 0; i < rings; i++)
            {
                for (int j = 0; j < segments; j++)
                {
                    int k1 = baseIndex + (i * (segments + 1)) + j;
                    int k2 = k1 + segments + 1;

                    if (i != 0)
                    {
                        indices.Add(new Vec3i(k1, k2, k1 + 1));
                        triangleMaterialIds.Add(materialId);
                    }
                    if (i != rings - 1)
                    {
                        indices.Add(new Vec3i(k1 + 1, k2, k2 + 1));
                        triangleMaterialIds.Add(materialId);
                    }
                }
            }

            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
        }

        // Axis-aligned box - 6 flat-shaded quads, one per face.
        public void AddBoxMesh(Vec3 min, Vec3 max, uint materialId)
        {
            Vec3 p000 = new Vec3(min.x, min.y, min.z), p100 = new Vec3(max.x, min.y, min.z);
            Vec3 p010 = new Vec3(min.x, max.y, min.z), p110 = new Vec3(max.x, max.y, min.z);
            Vec3 p001 = new Vec3(min.x, min.y, max.z), p101 = new Vec3(max.x, min.y, max.z);
            Vec3 p011 = new Vec3(min.x, max.y, max.z), p111 = new Vec3(max.x, max.y, max.z);

            AddQuad(p100, p110, p111, p101, new Vec3(1f, 0f, 0f), materialId);  // +X
            AddQuad(p000, p001, p011, p010, new Vec3(-1f, 0f, 0f), materialId); // -X
            AddQuad(p010, p011, p111, p110, new Vec3(0f, 1f, 0f), materialId);  // +Y
            AddQuad(p000, p100, p101, p001, new Vec3(0f, -1f, 0f), materialId); // -Y
            AddQuad(p001, p101, p111, p011, new Vec3(0f, 0f, 1f), materialId);  // +Z
            AddQuad(p000, p010, p110, p100, new Vec3(0f, 0f, -1f), materialId); // -Z
        }

        // Y-axis cylinder - smooth-shaded side (per-vertex radial normals, like the old
        // analytic cylinder) plus optional flat end caps.
        public void AddCylinderMesh(Vec3 center, float radius, float yMin, float yMax, uint materialId, bool capped = true, int segments = 20)
        {
            int indexStart = indices.Count;
            int baseIndex = vertices.Count;

            for (int i = 0; i <= 1; i++)
            {
                float y = i == 0 ? yMin : yMax;
                for (int j = 0; j <= segments; j++)
                {
                    float phi = j / (float)segments * 2f * (float)Math.PI;
                    Vec3 radial = new Vec3((float)Math.Cos(phi), 0f, (float)Math.Sin(phi));
                    vertices.Add(center + new Vec3(radial.x * radius, y - center.y, radial.z * radius));
                    normals.Add(radial);
                    // V=0 at yMax (top), V=1 at yMin (bottom) - same V-flip reasoning
                    // as AddQuad's own comment (matches this sample's "V=0 means top
                    // of the image" convention).
                    texCoords.Add(new Vec2(j / (float)segments, 1f - i));
                    tangents.Add(new Vec3(-radial.z, 0f, radial.x));
                }
            }

            for (int j = 0; j < segments; j++)
            {
                int k1 = baseIndex + j;
                int k2 = baseIndex + segments + 1 + j;
                indices.Add(new Vec3i(k1, k2, k1 + 1));
                triangleMaterialIds.Add(materialId);
                indices.Add(new Vec3i(k1 + 1, k2, k2 + 1));
                triangleMaterialIds.Add(materialId);
            }

            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });

            if (capped)
            {
                AddDiskMesh(new Vec3(center.x, yMin, center.z), new Vec3(0f, -1f, 0f), radius, materialId, segments);
                AddDiskMesh(new Vec3(center.x, yMax, center.z), new Vec3(0f, 1f, 0f), radius, materialId, segments);
            }
        }

        // Flat disk (triangle fan) with an arbitrary facing normal.
        public void AddDiskMesh(Vec3 center, Vec3 normal, float radius, uint materialId, int segments = 24)
        {
            int indexStart = indices.Count;
            int baseIndex = vertices.Count;

            Vec3 helper = Math.Abs(normal.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f);
            Vec3 u = Vec3.unitVector(Vec3.cross(helper, normal));
            Vec3 v = Vec3.cross(normal, u);

            vertices.Add(center);
            normals.Add(normal);
            texCoords.Add(new Vec2(0.5f, 0.5f));
            tangents.Add(u);

            for (int j = 0; j <= segments; j++)
            {
                float angle = j / (float)segments * 2f * (float)Math.PI;
                float cos = (float)Math.Cos(angle), sin = (float)Math.Sin(angle);
                vertices.Add(center + (radius * ((cos * u) + (sin * v))));
                normals.Add(normal);
                texCoords.Add(new Vec2(0.5f + (0.5f * cos), 0.5f + (0.5f * sin)));
                tangents.Add(u);
            }

            for (int j = 0; j < segments; j++)
            {
                indices.Add(new Vec3i(baseIndex, baseIndex + 1 + j, baseIndex + 2 + j));
                triangleMaterialIds.Add(materialId);
            }

            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
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
            tangents.AddRange(mesh.Tangents);

            foreach (var tri in mesh.Indices)
            {
                indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                triangleMaterialIds.Add(materialId);
            }
            meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
        }

        // Adds pre-loaded mesh data with per-triangle material assignments (for complex
        // models like Sponza where different materials are used). materialIdMap should
        // have length equal to mesh.Materials.Length with pre-allocated SceneBuilder
        // material IDs.
        public void AddMeshWithPerTriangleMaterials(OBJModel mesh, uint[] materialIdMap)
        {
            int indexStart = indices.Count;
            int baseIndex = vertices.Count;

            vertices.AddRange(mesh.Vertices);
            normals.AddRange(mesh.Normals);
            texCoords.AddRange(mesh.TexCoords);
            tangents.AddRange(mesh.Tangents);

            for (int triIdx = 0; triIdx < mesh.Indices.Length; triIdx++)
            {
                Vec3i tri = mesh.Indices[triIdx];
                indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));

                uint objMaterialId = mesh.TriangleMaterialIds[triIdx];
                uint materialId = materialIdMap[objMaterialId];
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
            tangents.AddRange(mesh.Tangents);
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

        // NeeLights/NeeLightCdf/NeeLightAreaPdf are deliberately left at their
        // SceneData defaults here, not computed inline - Scenes/LightList.cs needs the
        // shared environment map's total power, which is scene-independent and only
        // known to SampleRenderer (loaded once at
        // startup), not to a scene builder function. SampleRenderer.SwitchToScene
        // calls LightList.Build itself, once per scene switch, right after this
        // returns.
        public SceneData Build() => new SceneData
        {
            Name = Name,
            Vertices = vertices.ToArray(),
            Normals = normals.ToArray(),
            TexCoords = texCoords.ToArray(),
            Tangents = tangents.ToArray(),
            Indices = indices.ToArray(),
            TriangleMaterialIds = triangleMaterialIds.ToArray(),
            MeshRanges = meshRanges.Count > 0 ? FillMeshRangeGaps(meshRanges, indices.Count) : Array.Empty<MeshRange>(),
            Materials = materials.ToArray(),
            MaterialTexturePaths = materialTexturePaths.ToArray(),
            MaterialNormalTexturePaths = materialNormalTexturePaths.ToArray(),
            MaterialOrmTexturePaths = materialOrmTexturePaths.ToArray(),
            Lights = lights.ToArray(),
            OrbitingLights = orbitingLights.ToArray(),
            PulsingLights = pulsingLights.ToArray(),
            AmbientColor = AmbientColor,
            AmbientIntensity = AmbientIntensity,
            BackgroundTop = BackgroundTop,
            BackgroundBottom = BackgroundBottom,
            EnvMapPath = EnvMapPath,
            EnvMapIntensity = EnvMapIntensity,
            CameraOrigin = CameraOrigin,
            CameraLookAt = CameraLookAt,
            CameraUp = CameraUp,
            CameraFovDeg = CameraFovDeg,
            CameraWorldScale = CameraWorldScale,
        };
    }
}
