// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDenoiserTiling.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Native;
using System;
using System.Collections.Generic;

namespace ILGPU.OptiX.Denoising
{
    /// <summary>
    /// One tile produced by <see cref="OptixDenoiserTiling.SplitImage"/> - mirrors
    /// OptixUtilDenoiserImageTile (optix_denoiser_tiling.h).
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixDenoiserImageTile
    {
        /// <summary>The tile's input image (window into the full input, including overlap).</summary>
        public OptixImage2D Input;

        /// <summary>The tile's output image (window into the full output).</summary>
        public OptixImage2D Output;

        /// <summary>The inputOffsetX to pass to <see cref="OptixDenoiser.Invoke"/> for this tile.</summary>
        public uint InputOffsetX;

        /// <summary>The inputOffsetY to pass to <see cref="OptixDenoiser.Invoke"/> for this tile.</summary>
        public uint InputOffsetY;
    }

    /// <summary>
    /// C# port of the SDK's header-only tiled-denoising utility
    /// (optix_denoiser_tiling.h) - denoises images larger than the resolution the
    /// denoiser was set up for by splitting them into overlapping tiles and running
    /// back-to-back <see cref="OptixDenoiser.Invoke"/> calls that reuse one scratch
    /// allocation. Keep any behavioral change in sync with the SDK header.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixDenoiserTiling
    {
        /// <summary>
        /// Returns the pixel stride in bytes for an image, deriving it from the
        /// pixel format when the image's own PixelStrideInBytes is zero (port of
        /// optixUtilGetPixelStride).
        /// </summary>
        public static uint GetPixelStride(in OptixImage2D image)
        {
            if (image.PixelStrideInBytes != 0)
                return image.PixelStrideInBytes;

            return image.Format switch
            {
                OptixPixelFormat.Half1 => 1 * sizeof(short),
                OptixPixelFormat.Half2 => 2 * sizeof(short),
                OptixPixelFormat.Half3 => 3 * sizeof(short),
                OptixPixelFormat.Half4 => 4 * sizeof(short),
                OptixPixelFormat.Float1 => 1 * sizeof(float),
                OptixPixelFormat.Float2 => 2 * sizeof(float),
                OptixPixelFormat.Float3 => 3 * sizeof(float),
                OptixPixelFormat.Float4 => 4 * sizeof(float),
                OptixPixelFormat.UChar3 => 3,
                OptixPixelFormat.UChar4 => 4,
                _ => throw new OptixException(
                    OptixResult.OPTIX_ERROR_INVALID_VALUE,
                    $"Pixel format {image.Format} has no implicit pixel stride."),
            };
        }

        /// <summary>
        /// Splits a full-resolution input/output image pair into overlapping tiles
        /// (port of optixUtilDenoiserSplitImage). The overlap window size comes from
        /// <see cref="OptixDenoiserSizes.OverlapWindowSizeInPixels"/>.
        /// </summary>
        public static List<OptixDenoiserImageTile> SplitImage(
            in OptixImage2D input,
            in OptixImage2D output,
            uint overlapWindowSizeInPixels,
            uint tileWidth,
            uint tileHeight)
        {
            if (tileWidth == 0 || tileHeight == 0)
                throw new ArgumentException("Tile dimensions must be non-zero.");

            var tiles = new List<OptixDenoiserImageTile>();

            uint inPixelStride = GetPixelStride(input);
            uint outPixelStride = GetPixelStride(output);

            int overlap = (int)overlapWindowSizeInPixels;
            int inpW = (int)Math.Min(tileWidth + 2 * overlapWindowSizeInPixels, input.Width);
            int inpH = (int)Math.Min(tileHeight + 2 * overlapWindowSizeInPixels, input.Height);
            int inpY = 0, copiedY = 0;

            int upscaleX = (int)(output.Width / input.Width);
            int upscaleY = (int)(output.Height / input.Height);

            do
            {
                int inputOffsetY = inpY == 0
                    ? 0
                    : Math.Max(overlap, inpH - ((int)input.Height - inpY));
                int copyY = inpY == 0
                    ? (int)Math.Min(input.Height, tileHeight + overlapWindowSizeInPixels)
                    : (int)Math.Min(tileHeight, input.Height - (uint)copiedY);

                int inpX = 0, copiedX = 0;
                do
                {
                    int inputOffsetX = inpX == 0
                        ? 0
                        : Math.Max(overlap, inpW - ((int)input.Width - inpX));
                    int copyX = inpX == 0
                        ? (int)Math.Min(input.Width, tileWidth + overlapWindowSizeInPixels)
                        : (int)Math.Min(tileWidth, input.Width - (uint)copiedX);

                    var tile = new OptixDenoiserImageTile
                    {
                        Input = new OptixImage2D
                        {
                            Data = input.Data
                                + (ulong)(inpY - inputOffsetY) * input.RowStrideInBytes
                                + (ulong)(inpX - inputOffsetX) * inPixelStride,
                            Width = (uint)inpW,
                            Height = (uint)inpH,
                            RowStrideInBytes = input.RowStrideInBytes,
                            PixelStrideInBytes = input.PixelStrideInBytes,
                            Format = input.Format,
                        },
                        Output = new OptixImage2D
                        {
                            Data = output.Data
                                + (ulong)(upscaleY * inpY) * output.RowStrideInBytes
                                + (ulong)(upscaleX * inpX) * outPixelStride,
                            Width = (uint)(upscaleX * copyX),
                            Height = (uint)(upscaleY * copyY),
                            RowStrideInBytes = output.RowStrideInBytes,
                            PixelStrideInBytes = output.PixelStrideInBytes,
                            Format = output.Format,
                        },
                        InputOffsetX = (uint)inputOffsetX,
                        InputOffsetY = (uint)inputOffsetY,
                    };
                    tiles.Add(tile);

                    inpX += inpX == 0 ? (int)(tileWidth + overlapWindowSizeInPixels) : (int)tileWidth;
                    copiedX += copyX;
                } while (inpX < (int)input.Width);

                inpY += inpY == 0 ? (int)(tileHeight + overlapWindowSizeInPixels) : (int)tileHeight;
                copiedY += copyY;
            } while (inpY < (int)input.Height);

            return tiles;
        }

        /// <summary>
        /// Runs the denoiser tiled over input layers larger than the tile size
        /// (port of optixUtilDenoiserInvokeTiled) - splits every layer and guide
        /// image into overlapping tiles and performs back-to-back
        /// <see cref="OptixDenoiser.Invoke"/> calls reusing the single scratch
        /// buffer. The denoiser must have been set up (<see cref="OptixDenoiser.Setup"/>)
        /// for (tileWidth + 2 * overlap) x (tileHeight + 2 * overlap).
        /// </summary>
        /// <param name="denoiser">The denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="parameters">The denoiser parameters.</param>
        /// <param name="denoiserState">The denoiser state buffer.</param>
        /// <param name="denoiserStateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="guideLayer">The full-resolution guide layer images (may be default).</param>
        /// <param name="layers">The full-resolution input/output layer(s).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <param name="overlapWindowSizeInPixels">
        /// From <see cref="OptixDenoiserSizes.OverlapWindowSizeInPixels"/>
        /// (<see cref="OptixDenoiser.ComputeMemoryResources"/>).
        /// </param>
        /// <param name="tileWidth">Maximum tile width (excluding overlap).</param>
        /// <param name="tileHeight">Maximum tile height (excluding overlap).</param>
        public static void InvokeTiled(
            this OptixDenoiser denoiser,
            IntPtr stream,
            OptixDenoiserParams parameters,
            IntPtr denoiserState,
            ulong denoiserStateSizeInBytes,
            in OptixDenoiserGuideLayer guideLayer,
            ReadOnlySpan<OptixDenoiserLayer> layers,
            IntPtr scratch,
            ulong scratchSizeInBytes,
            uint overlapWindowSizeInPixels,
            uint tileWidth,
            uint tileHeight)
        {
            if (denoiser == null)
                throw new ArgumentNullException(nameof(denoiser));
            if (layers.Length == 0)
                throw new ArgumentException("At least one layer is required.", nameof(layers));

            uint upscale =
                layers[0].PreviousOutput.Width == 2 * layers[0].Input.Width ? 2u : 1u;

            var tiles = new List<OptixDenoiserImageTile>[layers.Length];
            var prevTiles = new List<OptixDenoiserImageTile>?[layers.Length];
            for (int l = 0; l < layers.Length; l++)
            {
                tiles[l] = SplitImage(
                    layers[l].Input, layers[l].Output,
                    overlapWindowSizeInPixels, tileWidth, tileHeight);

                if (layers[l].PreviousOutput.Data != 0)
                {
                    // The previous output is input-only - split against itself.
                    prevTiles[l] = SplitImage(
                        layers[l].PreviousOutput, layers[l].PreviousOutput,
                        upscale * overlapWindowSizeInPixels,
                        upscale * tileWidth, upscale * tileHeight);
                }
            }

            List<OptixDenoiserImageTile>? albedoTiles = guideLayer.Albedo.Data != 0
                ? SplitImage(guideLayer.Albedo, guideLayer.Albedo,
                    overlapWindowSizeInPixels, tileWidth, tileHeight)
                : null;
            List<OptixDenoiserImageTile>? normalTiles = guideLayer.Normal.Data != 0
                ? SplitImage(guideLayer.Normal, guideLayer.Normal,
                    overlapWindowSizeInPixels, tileWidth, tileHeight)
                : null;
            List<OptixDenoiserImageTile>? flowTiles = guideLayer.Flow.Data != 0
                ? SplitImage(guideLayer.Flow, guideLayer.Flow,
                    overlapWindowSizeInPixels, tileWidth, tileHeight)
                : null;
            List<OptixDenoiserImageTile>? flowTrustTiles =
                guideLayer.FlowTrustworthiness.Data != 0
                ? SplitImage(guideLayer.FlowTrustworthiness, guideLayer.FlowTrustworthiness,
                    overlapWindowSizeInPixels, tileWidth, tileHeight)
                : null;
            List<OptixDenoiserImageTile>? internalGuideLayerTiles =
                guideLayer.PreviousOutputInternalGuideLayer.Data != 0 &&
                guideLayer.OutputInternalGuideLayer.Data != 0
                ? SplitImage(
                    guideLayer.PreviousOutputInternalGuideLayer,
                    guideLayer.OutputInternalGuideLayer,
                    upscale * overlapWindowSizeInPixels,
                    upscale * tileWidth, upscale * tileHeight)
                : null;

            var tileLayers = new OptixDenoiserLayer[layers.Length];
            for (int t = 0; t < tiles[0].Count; t++)
            {
                for (int l = 0; l < layers.Length; l++)
                {
                    var layer = new OptixDenoiserLayer
                    {
                        Input = tiles[l][t].Input,
                        Output = tiles[l][t].Output,
                        Type = layers[l].Type,
                    };
                    if (layers[l].PreviousOutput.Data != 0)
                        layer.PreviousOutput = prevTiles[l]![t].Input;
                    tileLayers[l] = layer;
                }

                var tileGuideLayer = default(OptixDenoiserGuideLayer);
                if (albedoTiles != null)
                    tileGuideLayer.Albedo = albedoTiles[t].Input;
                if (normalTiles != null)
                    tileGuideLayer.Normal = normalTiles[t].Input;
                if (flowTiles != null)
                    tileGuideLayer.Flow = flowTiles[t].Input;
                if (flowTrustTiles != null)
                    tileGuideLayer.FlowTrustworthiness = flowTrustTiles[t].Input;
                if (internalGuideLayerTiles != null)
                {
                    tileGuideLayer.PreviousOutputInternalGuideLayer =
                        internalGuideLayerTiles[t].Input;
                    tileGuideLayer.OutputInternalGuideLayer =
                        internalGuideLayerTiles[t].Output;
                }

                denoiser.Invoke(
                    stream,
                    parameters,
                    denoiserState,
                    denoiserStateSizeInBytes,
                    tileGuideLayer,
                    tileLayers,
                    scratch,
                    scratchSizeInBytes,
                    tiles[0][t].InputOffsetX,
                    tiles[0][t].InputOffsetY);
            }
        }
    }
}
