using ILGPU.Algorithms;
using ILGPU.OptiX.Cuda;

namespace Sample15
{
    /// <summary>
    /// Device-side importance sampling and evaluation for the HDRI environment map -
    /// shared by NextEventEstimation.cs (picking
    /// a direction toward the environment) and Rays/RadianceRay.cs's __miss__radiance
    /// (evaluating the environment's radiance/pdf in a BSDF-sampled ray's own miss
    /// direction). Both
    /// sides use the same equirectangular direction&lt;-&gt;uv mapping and the same
    /// texel-quantized pdf lookup, which is what makes their MIS weights consistent.
    /// </summary>
    internal static class EnvironmentMapSampling
    {
        // Direction (unit vector, +Y up) -> equirectangular (u, v) in [0, 1)x[0, 1].
        // theta = polar angle from +Y (0 at the top pole, pi at the bottom); phi =
        // azimuth around Y. Self-consistent with TexelToDirection below - rotation
        // is subtracted here and added back
        // there, so a positive rotation spins the visible environment content the
        // same direction for both the NEE-sampling and miss-evaluation sides.
        internal static void DirectionToUv(Vec3 dir, float rotation, out float u, out float v)
        {
            float theta = XMath.Acos(XMath.Clamp(dir.y, -1f, 1f));
            float phi = XMath.Atan2(dir.z, dir.x) - rotation;
            float rawU = (phi + XMath.PI) / (2f * XMath.PI);
            // Wrap into [0, 1) - rotation can push phi outside [-pi, pi) before this
            // point, and CudaTex2D-style texel indexing has no built-in wraparound.
            u = rawU - XMath.Floor(rawU);
            v = theta / XMath.PI;
        }

        internal static Vec3 TexelToDirection(int row, int col, int width, int height, float rotation)
        {
            float u = (col + 0.5f) / width;
            float v = (row + 0.5f) / height;
            float theta = v * XMath.PI;
            float phi = (u * 2f * XMath.PI) - XMath.PI + rotation;
            float sinTheta = XMath.Sin(theta);
            return new Vec3(sinTheta * XMath.Cos(phi), XMath.Cos(theta), sinTheta * XMath.Sin(phi));
        }

        // Converts a texel's pdf-with-respect-to-the-(u,v)-unit-square (EnvMapPdfUv,
        // precomputed host-side) to a solid-angle pdf via the equirectangular
        // projection's Jacobian: u,v in [0,1]^2 map theta=v*pi, phi=u*2*pi, so
        // dOmega = sin(theta) dtheta dphi = 2*pi^2*sin(theta) du dv.
        internal static float PdfUvToSolidAngle(float pdfUv, int row, int height)
        {
            float v = (row + 0.5f) / height;
            float sinTheta = XMath.Sin(v * XMath.PI);
            return sinTheta > 1e-6f ? pdfUv / (2f * XMath.PI * XMath.PI * sinTheta) : 0f;
        }

        // Binary search over the flat luminance-weighted CDF (Scenes/EnvironmentMap.cs)
        // for the first texel whose cumulative probability is >= u.
        internal unsafe static int SampleTexelIndex(LaunchParams launchParams, float u)
        {
            int lo = 0;
            int hi = (launchParams.EnvMapWidth * launchParams.EnvMapHeight) - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (launchParams.EnvMapCdf[mid] < u)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            return lo;
        }

        // Full importance-sampled draw: picks a texel proportional to its radiance,
        // returns its (texel-center) direction, radiance, and solid-angle pdf. Used by
        // NextEventEstimation for the environment-map light kind.
        internal unsafe static Vec3 Sample(LaunchParams launchParams, ref LCG rng, out Vec3 radiance, out float pdfSolidAngle)
        {
            int width = launchParams.EnvMapWidth;
            int idx = SampleTexelIndex(launchParams, rng.Next());
            int row = idx / width;
            int col = idx % width;

            radiance = launchParams.EnvMapPixels[idx] * launchParams.EnvMapIntensity;
            pdfSolidAngle = PdfUvToSolidAngle(launchParams.EnvMapPdfUv[idx], row, launchParams.EnvMapHeight);
            return TexelToDirection(row, col, width, launchParams.EnvMapHeight, launchParams.EnvMapRotation);
        }

        // Evaluates the environment map in an arbitrary (already-known) direction -
        // used by __miss__radiance when a BSDF-sampled ray escapes the scene. Nearest-
        // texel lookup (matching Sample's own texel-center granularity, not a
        // bilinear-filtered fetch) so the pdf this call reports is exactly the pdf a
        // NEE draw landing on the same texel would have reported too - the consistency
        // MIS's power heuristic depends on, at the cost of some blockiness on an
        // extreme close-up of the sky (never the case through a pinhole camera at any
        // of this sample's scales).
        internal unsafe static Vec3 Evaluate(LaunchParams launchParams, Vec3 dir, out float pdfSolidAngle)
        {
            DirectionToUv(dir, launchParams.EnvMapRotation, out float u, out float v);
            int width = launchParams.EnvMapWidth;
            int height = launchParams.EnvMapHeight;
            int col = XMath.Clamp((int)(u * width), 0, width - 1);
            int row = XMath.Clamp((int)(v * height), 0, height - 1);
            int idx = (row * width) + col;

            pdfSolidAngle = PdfUvToSolidAngle(launchParams.EnvMapPdfUv[idx], row, height);
            return launchParams.EnvMapPixels[idx] * launchParams.EnvMapIntensity;
        }
    }
}
