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

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;

namespace Sample18
{
    /// <summary>
    /// Minimal position/index-only Wavefront OBJ loader (no materials/normals/UVs -
    /// Sample16 only needs real triangle counts/topology to exercise acceleration
    /// structure relocation against something more representative than a single quad,
    /// not a full shading pipeline). Ported from the same from-scratch parser used by
    /// Sample07/13-15's own Model.cs, trimmed down.
    /// </summary>
    public static class Model
    {
        public static (Vector3[] Vertices, Tri[] Indices) LoadPositionsOnly(string objPath)
        {
            var positions = new List<Vector3>();
            var indices = new List<Tri>();

            foreach (var rawLine in File.ReadLines(objPath))
            {
                var line = rawLine.Trim();
                if (line.Length == 0 || line[0] == '#')
                    continue;

                var tokens = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (tokens.Length == 0)
                    continue;

                switch (tokens[0])
                {
                    case "v":
                        positions.Add(new Vector3(
                            ParseFloat(tokens[1]),
                            ParseFloat(tokens[2]),
                            ParseFloat(tokens[3])));
                        break;

                    case "f":
                        {
                            // Fan-triangulate faces with more than 3 corners.
                            uint firstIndex = ResolveIndex(tokens[1], positions.Count);
                            uint prevIndex = ResolveIndex(tokens[2], positions.Count);
                            for (int i = 3; i < tokens.Length; i++)
                            {
                                uint currIndex = ResolveIndex(tokens[i], positions.Count);
                                indices.Add(new Tri(firstIndex, prevIndex, currIndex));
                                prevIndex = currIndex;
                            }
                            break;
                        }
                }
            }

            return (positions.ToArray(), indices.ToArray());
        }

        // Face vertex tokens are "v", "v/vt", "v//vn" or "v/vt/vn" - only the leading
        // position index is needed here.
        private static uint ResolveIndex(string token, int positionCount)
        {
            var slash = token.IndexOf('/');
            var vToken = slash < 0 ? token : token.Substring(0, slash);
            int idx = int.Parse(vToken, CultureInfo.InvariantCulture);
            // OBJ indices are 1-based, or negative to mean "relative to the current count".
            return (uint)(idx < 0 ? positionCount + idx : idx - 1);
        }

        private static float ParseFloat(string token) =>
            float.Parse(token, CultureInfo.InvariantCulture);
    }
}
