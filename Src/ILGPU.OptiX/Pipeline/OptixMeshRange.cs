// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixMeshRange.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// A sub-range of triangles within a shared index buffer - IndexStart/IndexCount are
    /// in triangle (index-triplet) units, not raw index count. Lets a scene with multiple
    /// meshes share one big vertex/index buffer as separate
    /// <see cref="AccelStructures.OptixAccelBuilder.AddTriangleMesh"/> build inputs, each
    /// covering only its own sub-range, instead of every build input covering the whole
    /// shared buffer.
    /// </summary>
    public struct OptixMeshRange
    {
        public int IndexStart;
        public int IndexCount;
    }

    /// <summary>
    /// Resolves a scene's triangle geometry into per-mesh <see cref="OptixMeshRange"/>s.
    /// Extracted from what used to be an identical, independently-hand-rolled
    /// "SbtLayout.GetTriangleMeshRanges" per sample (Sample13/14/15) - the same ~15 lines
    /// of boilerplate copy-pasted three times.
    /// </summary>
    public static class OptixSbtLayout
    {
        /// <summary>
        /// Returns <paramref name="explicitMeshRanges"/> as-is if
        /// <paramref name="useMergedTrianglesGas"/> is false and explicit ranges were
        /// tracked; otherwise returns a single range spanning the whole index buffer
        /// (the common case for scenes that don't track per-mesh boundaries, or that
        /// deliberately want everything built as one merged GAS build input).
        /// </summary>
        public static OptixMeshRange[] GetTriangleMeshRanges(
            int totalIndexCount, OptixMeshRange[] explicitMeshRanges, bool useMergedTrianglesGas) =>
            (!useMergedTrianglesGas && explicitMeshRanges != null && explicitMeshRanges.Length > 0)
                ? explicitMeshRanges
                : new[] { new OptixMeshRange { IndexStart = 0, IndexCount = totalIndexCount } };
    }
}
