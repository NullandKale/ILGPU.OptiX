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

namespace Sample10
{
    public class OBJMaterial
    {
        public string Name = string.Empty;
        public Vec3 Diffuse = new Vec3(0.8f, 0.8f, 0.8f);

        // Relative to the .obj's directory, or null if the material has no diffuse map.
        public string? DiffuseTexturePath;
    }

    // A minimal, from-scratch Wavefront OBJ/MTL loader (no external dependency - the
    // course's C++ reference uses tinyobjloader, but the format itself is simple enough
    // that hand-rolling it here avoids pulling in a new NuGet dependency for one sample).
    //
    // Unlike the C++ reference (one Mesh/GAS-build-input/SBT-record per material), this
    // still uses ONE merged vertex/index buffer for the whole model (simpler, no
    // per-mesh vertex duplication) - but per-material data (diffuse color, texture) is
    // looked up via optixGetSbtDataPointer against ONE hitgroup record per material per
    // ray type, same as the reference, using TriangleMaterialIds as the GAS build
    // input's SbtIndexOffsetBuffer (see SampleRenderer.cs's buildAccel/constructor).
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
                            if (File.Exists(mtlPath))
                            {
                                foreach (var material in LoadMaterials(mtlPath))
                                {
                                    materialsByName[material.Name] = material;
                                    materialOrder.Add(material);
                                }
                            }
                            break;
                        }

                    case "usemtl":
                        {
                            var name = tokens[1];
                            currentMaterialId = materialsByName.TryGetValue(name, out var material)
                                ? materialOrder.IndexOf(material)
                                : 0;
                            break;
                        }

                    case "v":
                        positions.Add(new Vec3(
                            ParseFloat(tokens[1]),
                            ParseFloat(tokens[2]),
                            ParseFloat(tokens[3])));
                        break;

                    case "vt":
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

            var model = new OBJModel
            {
                Vertices = combinedVertices.ToArray(),
                Normals = combinedNormals.ToArray(),
                TexCoords = combinedTexCoords.ToArray(),
                Indices = combinedIndices.ToArray(),
                TriangleMaterialIds = combinedMaterialIds.ToArray(),
                Materials = materialOrder.ToArray()
            };
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
                            yield return current;
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
                            current.DiffuseTexturePath = tokens[tokens.Length - 1];
                        break;
                }
            }
            if (current != null)
                yield return current;
        }
    }
}
