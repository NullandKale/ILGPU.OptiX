// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: InputEncoding.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms;
using ILGPU.OptiX.Device;
using Sample21.Core;
using Half = ILGPU.Half;

namespace Sample21.Device.Network
{
    /// <summary>
    /// Ports VkNRC's <c>NRCInputEncode</c> (example/VkNRC/shader/src/NRCRecord.glsl)
    /// faithfully at the default width (64): 3x12-band triangle-wave frequency encoding
    /// on world position (36), one-blob-4 on the spherical-UV-encoded scattered
    /// direction (8) and shading normal (8), one-blob-4 on remapped roughness (4), raw
    /// diffuse+specular reflectance (6, +2 padding ones) - 64 values total, matching
    /// <see cref="NrcConstants.LayerWidth"/>. Positions are expected pre-scaled into a
    /// bounded, roughly [-1,1] scene-local range by the caller (VkNRC does the same -
    /// the frequency bands top out at 2048x, so an unbounded world-space position would
    /// alias badly).
    ///
    /// <para>
    /// <see cref="Encode"/> is runtime-parameterized by <c>targetWidth</c> - unlike
    /// <see cref="ForwardKernel"/>/<see cref="BackwardKernel"/>, this method makes no
    /// OptixCoopVec calls (plain memory writes only), so width is an ordinary parameter
    /// here, not a PTX compile-time constant. Every category except the position
    /// frequency bands stays fixed (8 dir-blob + 8 normal-blob + 4 roughness-blob + 6 raw
    /// = <see cref="FixedSlotCount"/>); the position bands scale to fill
    /// <c>targetWidth - FixedSlotCount</c> evenly across the 3 axes, with any remainder
    /// (from non-multiple-of-3 widths) appended as constant-1 padding - this exact
    /// formula reproduces VkNRC's original 12-band/2-padding split unchanged at
    /// targetWidth=64 (see <see cref="Encode"/>'s own doc comment).
    /// </para>
    /// </summary>
    public static class InputEncoding
    {
        private const float TwoPi = 6.28318530717959f;

        /// <summary>Non-position-band slots: 4 (dir U) + 4 (dir V) + 4 (normal U) + 4 (normal V) + 4 (roughness) + 6 (raw diffuse/specular) = 26.</summary>
        private const int FixedSlotCount = 26;

        // A cheap triangle-wave standing in for sin(pi*x) - same choice VkNRC makes
        // (NRCRecord.glsl's _nrc_tri, with the sin() alternative commented out there
        // too) since it's several ALU ops cheaper per band and empirically
        // interchangeable for this purpose.
        private static float Tri(float x)
        {
            float m = x - (2f * XMath.Floor(x / 2f)); // x mod 2, always >= 0
            return (2f * XMath.Abs(m - 1f)) - 1f;
        }

        // Scalar form of VkNRC's _quartic_cdf(x, inv_radius=4) - a quartic smoothstep-like
        // CDF used to build the one-blob (soft one-hot) encoding below.
        private static float QuarticCdf(float x)
        {
            float u = x * 4f;
            float u2 = u * u;
            float u4 = u2 * u2;
            return XMath.Clamp(((15f / 16f) * u * (1f - ((2f / 3f) * u2) + ((1f / 5f) * u4))) + 0.5f, 0f, 1f);
        }

        // One-blob-4: a soft one-hot encoding of x in [0,1] across 4 bins - VkNRC's
        // NRCOneBlob4Encode, unrolled to 4 scalar outputs (device code has no convenient
        // small-vector return here the way GLSL's vec4 does).
        private static void OneBlob4(float x, out float b0, out float b1, out float b2, out float b3)
        {
            b0 = QuarticCdf(0.25f - x) - QuarticCdf(0f - x);
            b1 = QuarticCdf(0.5f - x) - QuarticCdf(0.25f - x);
            b2 = QuarticCdf(0.75f - x) - QuarticCdf(0.5f - x);
            b3 = QuarticCdf(1f - x) - QuarticCdf(0.75f - x);
        }

        // VkNRC's NRCSphEncode - world-space unit direction -> [0,1]^2 spherical UV.
        private static void SphEncode(Vec3 d, out float u, out float v)
        {
            u = (d.x == 0f && d.y == 0f) ? 0.5f : 0.5f + (XMath.Atan2(d.y, d.x) / TwoPi);
            v = XMath.Acos(XMath.Clamp(d.z, -1f, 1f)) / XMath.PI;
        }

        // Frequency-encode one scalar in [0,1]-ish range (position axis, pre-scaled by
        // the caller) across bandCount consecutive Half slots starting at
        // actScratch[baseIdx + offset] - bandCount is 12 at the default width=64 (see
        // Encode's own doc comment for the derivation at other widths).
        private static void FrequencyEncodeN(OptixDeviceView<Half> actScratch, long baseIdx, int offset, float p, int bandCount)
        {
            float freq = 1f;
            for (int i = 0; i < bandCount; i++)
            {
                actScratch[baseIdx + offset + i] = (Half)Tri(freq * p);
                freq *= 2f;
            }
        }

        /// <summary>
        /// Writes the full <paramref name="targetWidth"/>-wide encoded input for record
        /// <paramref name="recordIdx"/> into slot 0 of its act-scratch region
        /// (<paramref name="actScratch"/> is the same dual-access buffer the training
        /// kernels address by raw device pointer for CoopVec calls). <paramref name="slotsPerRecord"/> is the
        /// per-record act-scratch stride in <paramref name="targetWidth"/>-wide slots:
        /// <c>hiddenLayerCount + 1</c> for the training path (every layer's activation
        /// cached for backprop) or <c>2</c> for the forward-only inference path's
        /// ping-pong slots (INrcNetworkOps.ForwardInference). At
        /// <paramref name="targetWidth"/>=64 this produces VkNRC's original layout
        /// (12-band position, 2-slot padding) - see this class's own doc comment for
        /// the band-count-scaling formula at other widths.
        /// </summary>
        public static void Encode(
            OptixDeviceView<Half> actScratch,
            long recordIdx,
            Vec3 position,
            Vec3 scatteredDir,
            Vec3 normal,
            float roughness,
            Vec3 diffuse,
            Vec3 specular,
            int slotsPerRecord,
            int targetWidth)
        {
            long baseIdx = recordIdx * slotsPerRecord * targetWidth;

            int bandsPerAxis = (targetWidth - FixedSlotCount) / 3;
            int padCount = targetWidth - FixedSlotCount - (3 * bandsPerAxis);

            FrequencyEncodeN(actScratch, baseIdx, 0, position.x, bandsPerAxis);
            FrequencyEncodeN(actScratch, baseIdx, bandsPerAxis, position.y, bandsPerAxis);
            FrequencyEncodeN(actScratch, baseIdx, 2 * bandsPerAxis, position.z, bandsPerAxis);
            int o = 3 * bandsPerAxis;

            SphEncode(scatteredDir, out float su, out float sv);
            OneBlob4(su, out float s0, out float s1, out float s2, out float s3);
            actScratch[baseIdx + o + 0] = (Half)s0;
            actScratch[baseIdx + o + 1] = (Half)s1;
            actScratch[baseIdx + o + 2] = (Half)s2;
            actScratch[baseIdx + o + 3] = (Half)s3;
            o += 4;
            OneBlob4(sv, out float s4, out float s5, out float s6, out float s7);
            actScratch[baseIdx + o + 0] = (Half)s4;
            actScratch[baseIdx + o + 1] = (Half)s5;
            actScratch[baseIdx + o + 2] = (Half)s6;
            actScratch[baseIdx + o + 3] = (Half)s7;
            o += 4;

            SphEncode(normal, out float nu, out float nv);
            OneBlob4(nu, out float n0, out float n1, out float n2, out float n3);
            actScratch[baseIdx + o + 0] = (Half)n0;
            actScratch[baseIdx + o + 1] = (Half)n1;
            actScratch[baseIdx + o + 2] = (Half)n2;
            actScratch[baseIdx + o + 3] = (Half)n3;
            o += 4;
            OneBlob4(nv, out float n4, out float n5, out float n6, out float n7);
            actScratch[baseIdx + o + 0] = (Half)n4;
            actScratch[baseIdx + o + 1] = (Half)n5;
            actScratch[baseIdx + o + 2] = (Half)n6;
            actScratch[baseIdx + o + 3] = (Half)n7;
            o += 4;

            // VkNRC's own remap: 1 - exp(-roughness), squashes the unbounded
            // (though typically [0,1]-ish already) roughness into [0,1) before one-blob.
            float roughnessRemapped = 1f - XMath.Exp(-roughness);
            OneBlob4(roughnessRemapped, out float r0, out float r1, out float r2, out float r3);
            actScratch[baseIdx + o + 0] = (Half)r0;
            actScratch[baseIdx + o + 1] = (Half)r1;
            actScratch[baseIdx + o + 2] = (Half)r2;
            actScratch[baseIdx + o + 3] = (Half)r3;
            o += 4;

            actScratch[baseIdx + o + 0] = (Half)diffuse.x;
            actScratch[baseIdx + o + 1] = (Half)diffuse.y;
            actScratch[baseIdx + o + 2] = (Half)diffuse.z;
            actScratch[baseIdx + o + 3] = (Half)specular.x;
            actScratch[baseIdx + o + 4] = (Half)specular.y;
            actScratch[baseIdx + o + 5] = (Half)specular.z;
            o += 6;

            // Constant-1 padding for any remainder targetWidth isn't evenly divisible
            // into (see this class's own doc comment) - VkNRC's original 64-wide layout
            // is exactly this formula's targetWidth=64 case (padCount=2).
            for (int i = 0; i < padCount; i++)
                actScratch[baseIdx + o + i] = (Half)1f;
        }
    }
}
