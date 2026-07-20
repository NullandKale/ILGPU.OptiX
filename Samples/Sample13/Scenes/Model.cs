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

namespace Sample13
{
    public class OBJMaterial
    {
        public string Name = string.Empty;
        public Vec3 Diffuse = new Vec3(0.8f, 0.8f, 0.8f);

        // Relative to the .obj's directory, or null if the material has no diffuse map.
        public string? DiffuseTexturePath;
    }

    // A minimal, from-scratch Wavefront OBJ/MTL loader - one merged vertex/index
    // buffer for the whole model, per-material data looked up via
    // optixGetSbtDataPointer against one hitgroup record per material (see
    // SampleRenderer.cs), same convention as every other scene in this sample.
    public class OBJModel
    {
        public Vec3[] Vertices = Array.Empty<Vec3>();
        public Vec3[] Normals = Array.Empty<Vec3>();
        public Vec2[] TexCoords = Array.Empty<Vec2>();
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
                        // OBJ's vt convention is V=0 at the image's bottom row, but
                        // TextureLoader.LoadRgba8 (and the CUDA array it uploads into)
                        // store pixel data top-to-bottom, i.e. V=0 = the top row. Left
                        // unflipped, every OBJ-sourced texture sample (diffuse color
                        // and, critically, an alpha-cutout mask like Sponza's leaf
                        // texture) reads the wrong row - flip here once at load time so
                        // every consumer (SampleAlbedo/SampleAlpha) just works.
                        texcoords.Add(new Vec2(
                            ParseFloat(tokens[1]),
                            1f - ParseFloat(tokens[2])));
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

            // cow.obj/stanford-bunny.obj/teapot.obj/xyzrgb_dragon.obj are all confirmed
            // to have zero vn lines - without this fallback, every
            // vertex would keep ResolveVertex's default (0,1,0) normal below, which
            // visibly breaks shading (every triangle would light as if facing straight
            // up regardless of its actual orientation). Compute smooth per-vertex
            // normals instead whenever the file supplied none at all; a file that does
            // supply vn data (e.g. Sponza) keeps using it unchanged.
            var resolvedNormals = normals.Count == 0
                ? ComputeSmoothNormals(combinedVertices, combinedIndices)
                : combinedNormals.ToArray();

            var model = new OBJModel
            {
                Vertices = combinedVertices.ToArray(),
                Normals = resolvedNormals,
                TexCoords = combinedTexCoords.ToArray(),
                Indices = combinedIndices.ToArray(),
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

        // Area-weighted (unnormalized face-normal accumulation, so larger adjacent
        // triangles contribute more) smooth per-vertex normals. Vertices are already
        // deduped by (posIdx,vtIdx,vnIdx) in ResolveVertex - since a vn-less file has no
        // vt/vn either for these meshes, every occurrence of the same position collapses
        // to one shared vertex, so this naturally accumulates over every triangle
        // touching that position.
        private static Vec3[] ComputeSmoothNormals(List<Vec3> vertices, List<Vec3i> indices)
        {
            var accum = new Vec3[vertices.Count];
            foreach (var tri in indices)
            {
                Vec3 a = vertices[tri.x];
                Vec3 b = vertices[tri.y];
                Vec3 c = vertices[tri.z];
                Vec3 faceNormal = Vec3.cross(b - a, c - a);
                accum[tri.x] += faceNormal;
                accum[tri.y] += faceNormal;
                accum[tri.z] += faceNormal;
            }

            var result = new Vec3[vertices.Count];
            for (int i = 0; i < result.Length; i++)
            {
                float lengthSq = accum[i].lengthSquared();
                result[i] = lengthSq > 1e-12f ? Vec3.unitVector(accum[i]) : new Vec3(0f, 1f, 0f);
            }
            return result;
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
