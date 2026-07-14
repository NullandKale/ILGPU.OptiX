// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Model.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

#nullable enable

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ILGPU.OptiX.Cuda;

namespace Sample15
{
    public class OBJMaterial
    {
        public string Name = string.Empty;
        public Vec3 Diffuse = new Vec3(0.8f, 0.8f, 0.8f);

        // Relative to the .obj's directory, or null if the material has no diffuse map.
        public string? DiffuseTexturePath;

        // PBR extensions - none of this repo's bundled MTLs (confirmed: Sponza's)
        // actually populate these, so they're parsed correctly but exercised only
        // once a scene supplies an asset that has them. RoughnessTexturePath/
        // MetallicTexturePath are parsed for completeness but not wired to any
        // MaterialSbtData field - this sample's OrmTexture expects one already-packed
        // (occlusion.r/roughness.g/metallic.b) texture (an artist/tool convention, not
        // something reconstructed from separate map_Pr/map_Pm grayscale maps at load
        // time - no bundled asset needs that composition step, so it isn't built).
        public string? NormalTexturePath;
        public string? RoughnessTexturePath;
        public string? MetallicTexturePath;
        public float? Roughness;
        public float? Metallic;
        public Vec3 Emission = new Vec3(0f, 0f, 0f);
    }

    // A minimal, from-scratch Wavefront OBJ/MTL loader, ported from Sample07's Model.cs -
    // one merged vertex/index buffer for the whole model, per-material data looked up via
    // optixGetSbtDataPointer against one hitgroup record per material (see
    // SampleRenderer.cs), same convention as every other scene in this sample.
    public class OBJModel
    {
        public Vec3[] Vertices = Array.Empty<Vec3>();
        public Vec3[] Normals = Array.Empty<Vec3>();
        public Vec2[] TexCoords = Array.Empty<Vec2>();
        // Per-vertex tangent - always computed (ComputeTangents), independent of
        // whether any material on this model
        // actually has a normal map; unused device-side unless MaterialSbtData.
        // NormalTexture is nonzero for the triangle being shaded.
        public Vec3[] Tangents = Array.Empty<Vec3>();
        public Vec3i[] Indices = Array.Empty<Vec3i>();

        // One entry per triangle (same length as Indices), indexing into Materials.
        public uint[] TriangleMaterialIds = Array.Empty<uint>();
        public OBJMaterial[] Materials = Array.Empty<OBJMaterial>();

        public static OBJModel Load(string objPath)
        {
            Console.WriteLine($"[OBJ] Loading: {objPath}");
            string baseDir = Path.GetDirectoryName(Path.GetFullPath(objPath)) ?? ".";

            var materialsByName = new Dictionary<string, OBJMaterial>(StringComparer.Ordinal);
            var materialOrder = new List<OBJMaterial>();
            var defaultMaterial = new OBJMaterial { Name = string.Empty };
            materialOrder.Add(defaultMaterial);

            var positions = new List<Vec3>();
            var texcoords = new List<Vec2>();
            var normals = new List<Vec3>();

            var combinedVertices = new List<Vec3>();
            var combinedNormals = new List<Vec3>();
            var combinedTexCoords = new List<Vec2>();
            var combinedIndices = new List<Vec3i>();
            var combinedMaterialIds = new List<uint>();

            var vertexCache = new Dictionary<(int, int, int), int>();
            int currentMaterialId = 0;

            foreach (var rawLine in File.ReadLines(objPath))
            {
                var line = rawLine.Trim();
                if (line.Length == 0 || line[0] == '#')
                    continue;

                var tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length == 0)
                    continue;

                switch (tokens[0])
                {
                    case "mtllib":
                        {
                            var mtlPath = Path.Combine(baseDir, tokens[1]);
                            Console.WriteLine($"[OBJ] Found mtllib: {tokens[1]}");
                            if (File.Exists(mtlPath))
                            {
                                var loadedMaterials = LoadMaterials(mtlPath);
                                Console.WriteLine($"[OBJ] Loaded materials from {mtlPath}");
                                foreach (var material in loadedMaterials)
                                {
                                    Console.WriteLine($"[OBJ]   Material: '{material.Name}' Diffuse=({material.Diffuse.x:F2},{material.Diffuse.y:F2},{material.Diffuse.z:F2}) Texture={material.DiffuseTexturePath ?? "(none)"}");
                                    materialsByName[material.Name] = material;
                                    materialOrder.Add(material);
                                }
                            }
                            else
                            {
                                Console.WriteLine($"[OBJ] WARNING: MTL file not found: {mtlPath}");
                            }
                            break;
                        }

                    case "usemtl":
                        {
                            var name = tokens[1];
                            currentMaterialId = materialsByName.TryGetValue(name, out var material)
                                ? materialOrder.IndexOf(material)
                                : 0;
                            Console.WriteLine($"[OBJ] Using material: '{name}' (id={currentMaterialId})");
                            break;
                        }

                    case "v":
                        positions.Add(new Vec3(
                            ParseFloat(tokens[1]),
                            ParseFloat(tokens[2]),
                            ParseFloat(tokens[3])));
                        break;

                    case "vt":
                        // No V-flip: this bundled Sponza .obj's own "vt" v-values are
                        // already authored against a top-left texture origin (matching
                        // TextureLoader.LoadRgba8/CudaTextureObject's own row 0 = top
                        // convention), not the raw OBJ-spec bottom-left origin the
                        // "1f - v" flip previously here assumed in theory. That flip
                        // was carried over unchanged from Sample07 through Sample14
                        // (identical code, confirmed by diff) and never actually
                        // visually verified for orientation until now - user-reported
                        // "lion head and curtain textures in sponza are upside down
                        // compared to the geometry" with the flip in place, and no
                        // other place in the pipeline touches V (SampleAlbedo/
                        // SampleAlpha/SampleOrm/ApplyNormalMap all use the interpolated
                        // uv.y as-is - see MaterialShading.cs), so the flip itself was
                        // the bug. If a *different* OBJ asset is ever added that
                        // genuinely does follow the spec's bottom-left convention, it
                        // will need its own per-asset flip instead of a global one here.
                        texcoords.Add(new Vec2(
                            ParseFloat(tokens[1]),
                            ParseFloat(tokens[2])));
                        break;

                    case "vn":
                        normals.Add(new Vec3(
                            ParseFloat(tokens[1]),
                            ParseFloat(tokens[2]),
                            ParseFloat(tokens[3])));
                        break;

                    case "f":
                        {
                            // Fan-triangulate faces with more than 3 corners.
                            int firstIndex = ResolveVertex(tokens[1]);
                            int prevIndex = ResolveVertex(tokens[2]);
                            for (int i = 3; i < tokens.Length; i++)
                            {
                                int currIndex = ResolveVertex(tokens[i]);
                                combinedIndices.Add(new Vec3i(firstIndex, prevIndex, currIndex));
                                combinedMaterialIds.Add((uint)currentMaterialId);
                                prevIndex = currIndex;
                            }
                            break;
                        }
                }
            }

            // cow.obj/stanford-bunny.obj/teapot.obj/xyzrgb_dragon.obj (M6's mesh scenes)
            // are all confirmed to have zero vn lines - without this fallback, every
            // vertex would keep ResolveVertex's default (0,1,0) normal below, which
            // visibly breaks shading (every triangle would light as if facing straight
            // up regardless of its actual orientation). Compute smooth per-vertex
            // normals instead whenever the file supplied none at all; a file that does
            // supply vn data (e.g. Sponza) keeps using it unchanged.
            var resolvedNormals = normals.Count == 0
                ? ComputeSmoothNormals(combinedVertices, combinedIndices)
                : combinedNormals.ToArray();

            var resolvedTexCoords = combinedTexCoords.ToArray();
            var resolvedIndices = combinedIndices.ToArray();
            var model = new OBJModel
            {
                Vertices = combinedVertices.ToArray(),
                Normals = resolvedNormals,
                TexCoords = resolvedTexCoords,
                Tangents = ComputeTangents(combinedVertices, resolvedNormals, resolvedTexCoords, resolvedIndices),
                Indices = resolvedIndices,
                TriangleMaterialIds = combinedMaterialIds.ToArray(),
                Materials = materialOrder.ToArray()
            };

            Console.WriteLine($"[OBJ] Load complete: {model.Vertices.Length} vertices, {model.Indices.Length} triangles, {model.Materials.Length} materials");
            for (int i = 0; i < model.Materials.Length; i++)
            {
                var mat = model.Materials[i];
                Console.WriteLine($"[OBJ]   Material {i}: '{mat.Name}' Texture={mat.DiffuseTexturePath ?? "(none)"}");
            }

            return model;

            int ResolveVertex(string token)
            {
                var parts = token.Split('/');
                int vIdx = ResolveObjIndex(parts[0], positions.Count);
                int vtIdx = parts.Length > 1 && parts[1].Length > 0
                    ? ResolveObjIndex(parts[1], texcoords.Count)
                    : -1;
                int vnIdx = parts.Length > 2 && parts[2].Length > 0
                    ? ResolveObjIndex(parts[2], normals.Count)
                    : -1;

                var key = (vIdx, vtIdx, vnIdx);
                if (vertexCache.TryGetValue(key, out var existing))
                    return existing;

                int newIndex = combinedVertices.Count;
                combinedVertices.Add(positions[vIdx]);
                combinedNormals.Add(vnIdx >= 0 ? normals[vnIdx] : new Vec3(0, 1, 0));
                combinedTexCoords.Add(vtIdx >= 0 ? texcoords[vtIdx] : new Vec2(0, 0));
                vertexCache[key] = newIndex;
                return newIndex;
            }
        }

        // Angle-weighted (not area-weighted) smooth per-vertex normals - each
        // triangle's *normalized* face normal is scaled by the subtended angle at
        // each of its own three vertices before accumulating, rather than by the
        // triangle's raw (unnormalized-cross-product) area. Area-weighting lets a
        // handful of large, thin, near-degenerate triangles dominate a vertex's
        // normal regardless of the actual surface angle there - exactly the failure
        // mode on thin, elongated geometry like the Stanford Bunny's ears (no vn data
        // to fall back to - see Load()), where it produced visibly wavy GGX specular
        // highlights instead of smooth curves, noticed once the M2/M7 material sweep
        // scene put several roughness-varying Bunny copies under sharp specular
        // lighting for the first time. Angle-weighting is the standard fix (Max, N.
        // "Weights for Computing Vertex Normals from Facet Vectors", 1999) but doesn't
        // eliminate linear barycentric normal interpolation's own inherent waviness on
        // curved surfaces - only finer tessellation or a higher-order interpolation
        // scheme would do that, both out of scope here. Vertices are already deduped by
        // (posIdx,vtIdx,vnIdx) in ResolveVertex - since a vn-less file has no vt/vn
        // either for these meshes, every occurrence of the same position collapses to
        // one shared vertex, so this naturally accumulates over every triangle
        // touching that position.
        private static Vec3[] ComputeSmoothNormals(List<Vec3> vertices, List<Vec3i> indices)
        {
            var accum = new Vec3[vertices.Count];
            foreach (var tri in indices)
            {
                Vec3 a = vertices[tri.x];
                Vec3 b = vertices[tri.y];
                Vec3 c = vertices[tri.z];
                Vec3 rawFaceNormal = Vec3.cross(b - a, c - a);
                if (rawFaceNormal.lengthSquared() <= 1e-20f)
                    continue; // degenerate (zero-area) triangle - contributes nothing
                Vec3 faceNormal = Vec3.unitVector(rawFaceNormal);

                accum[tri.x] += faceNormal * VertexAngle(a, b, c);
                accum[tri.y] += faceNormal * VertexAngle(b, c, a);
                accum[tri.z] += faceNormal * VertexAngle(c, a, b);
            }

            var result = new Vec3[vertices.Count];
            for (int i = 0; i < result.Length; i++)
            {
                float lengthSq = accum[i].lengthSquared();
                result[i] = lengthSq > 1e-12f ? Vec3.unitVector(accum[i]) : new Vec3(0f, 1f, 0f);
            }
            return result;
        }

        // The interior angle of triangle (at, b, c) measured at vertex `at`.
        private static float VertexAngle(Vec3 at, Vec3 b, Vec3 c)
        {
            Vec3 u = Vec3.unitVector(b - at);
            Vec3 v = Vec3.unitVector(c - at);
            float cosAngle = Math.Clamp(Vec3.dot(u, v), -1f, 1f);
            return MathF.Acos(cosAngle);
        }

        // Per-triangle UV-gradient tangent solve (Lengyel's formula), area-weighted-
        // accumulated per vertex (same shape as ComputeSmoothNormals - the raw,
        // unnormalized per-triangle tangent already scales with triangle area since
        // it's built from the unnormalized edge vectors), then Gram-Schmidt-
        // orthogonalized against each vertex's own final shading normal so the result
        // is always perpendicular to it regardless of which normal source (vn data or
        // ComputeSmoothNormals' fallback) was used. A degenerate UV parameterization
        // (duplicate/collapsed texcoords, or a
        // model with no vt data at all) falls back to an arbitrary tangent orthogonal
        // to the normal - never a zero vector, so device-side TBN construction never
        // divides by zero.
        private static Vec3[] ComputeTangents(List<Vec3> vertices, Vec3[] normals, Vec2[] texCoords, Vec3i[] indices)
        {
            var accum = new Vec3[vertices.Count];
            foreach (var tri in indices)
            {
                Vec3 v0 = vertices[tri.x], v1 = vertices[tri.y], v2 = vertices[tri.z];
                Vec2 uv0 = texCoords[tri.x], uv1 = texCoords[tri.y], uv2 = texCoords[tri.z];

                Vec3 e1 = v1 - v0, e2 = v2 - v0;
                float du1 = uv1.x - uv0.x, dv1 = uv1.y - uv0.y;
                float du2 = uv2.x - uv0.x, dv2 = uv2.y - uv0.y;

                float denom = (du1 * dv2) - (du2 * dv1);
                if (MathF.Abs(denom) <= 1e-12f)
                    continue;
                float r = 1f / denom;
                Vec3 tangent = ((e1 * dv2) - (e2 * dv1)) * r;

                accum[tri.x] += tangent;
                accum[tri.y] += tangent;
                accum[tri.z] += tangent;
            }

            var result = new Vec3[vertices.Count];
            for (int i = 0; i < result.Length; i++)
            {
                Vec3 n = normals[i];
                Vec3 t = accum[i];
                // Gram-Schmidt: remove any component of t along n, then normalize.
                t -= n * Vec3.dot(n, t);
                float lengthSq = t.lengthSquared();
                result[i] = lengthSq > 1e-12f ? Vec3.unitVector(t) : ArbitraryOrthogonal(n);
            }
            return result;
        }

        private static Vec3 ArbitraryOrthogonal(Vec3 n)
        {
            Vec3 up = MathF.Abs(n.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f);
            return Vec3.unitVector(Vec3.cross(up, n));
        }

        // OBJ indices are 1-based, or negative to mean "relative to the current count".
        private static int ResolveObjIndex(string token, int count)
        {
            int idx = int.Parse(token, CultureInfo.InvariantCulture);
            return idx < 0 ? count + idx : idx - 1;
        }

        private static float ParseFloat(string token) =>
            float.Parse(token, CultureInfo.InvariantCulture);

        private static IEnumerable<OBJMaterial> LoadMaterials(string mtlPath)
        {
            Console.WriteLine($"[MTL] Loading: {mtlPath}");
            OBJMaterial? current = null;
            foreach (var rawLine in File.ReadLines(mtlPath))
            {
                var line = rawLine.Trim();
                if (line.Length == 0 || line[0] == '#')
                    continue;

                var tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length == 0)
                    continue;

                switch (tokens[0])
                {
                    case "newmtl":
                        if (current != null)
                        {
                            Console.WriteLine($"[MTL]   Material: '{current.Name}' Texture={current.DiffuseTexturePath ?? "(none)"}");
                            yield return current;
                        }
                        current = new OBJMaterial { Name = tokens[1] };
                        break;

                    case "Kd":
                        if (current != null && tokens.Length >= 4)
                        {
                            current.Diffuse = new Vec3(
                                ParseFloat(tokens[1]),
                                ParseFloat(tokens[2]),
                                ParseFloat(tokens[3]));
                        }
                        break;

                    case "map_Kd":
                        if (current != null)
                        {
                            current.DiffuseTexturePath = tokens[tokens.Length - 1];
                            Console.WriteLine($"[MTL]   Found texture: {current.DiffuseTexturePath}");
                        }
                        break;

                    // map_Bump/norm are the two conventional MTL tag spellings for a
                    // tangent-space normal map.
                    case "map_Bump":
                    case "norm":
                        if (current != null)
                            current.NormalTexturePath = tokens[tokens.Length - 1];
                        break;

                    case "map_Pr":
                        if (current != null)
                            current.RoughnessTexturePath = tokens[tokens.Length - 1];
                        break;

                    case "map_Pm":
                        if (current != null)
                            current.MetallicTexturePath = tokens[tokens.Length - 1];
                        break;

                    case "Pr":
                        if (current != null && tokens.Length >= 2)
                            current.Roughness = ParseFloat(tokens[1]);
                        break;

                    case "Pm":
                        if (current != null && tokens.Length >= 2)
                            current.Metallic = ParseFloat(tokens[1]);
                        break;

                    case "Ke":
                        if (current != null && tokens.Length >= 4)
                        {
                            current.Emission = new Vec3(
                                ParseFloat(tokens[1]),
                                ParseFloat(tokens[2]),
                                ParseFloat(tokens[3]));
                        }
                        break;
                }
            }
            if (current != null)
            {
                Console.WriteLine($"[MTL]   Material: '{current.Name}' Texture={current.DiffuseTexturePath ?? "(none)"}");
                yield return current;
            }
        }
    }
}
