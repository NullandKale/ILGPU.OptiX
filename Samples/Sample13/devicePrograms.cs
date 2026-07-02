using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample13
{
    public static class devicePrograms
    {
        private const uint RADIANCE_RAY_TYPE = 0;
        private const uint SHADOW_RAY_TYPE = 1;
        private const uint RAY_TYPE_COUNT = 2;

        // Oren-Nayar roughness, matching the reference's fixed DiffuseSigmaDeg = 25.0f
        // (RayTracing/RaytraceRenderer.cs) - the reference has no per-material roughness.
        private const float DiffuseSigmaDeg = 25f;

        // Continuation flags returned via Payload3 by __closest__radiance - matches the
        // reference's MaxMirrorBounces/MaxRefractions caps (both 2), tracked as separate
        // counters in raygen's own loop (closesthit doesn't need to know the running
        // counts - it only reports which kind of surface was hit).
        private const uint BOUNCE_TERMINAL = 0;
        private const uint BOUNCE_CONTINUE_MIRROR = 1;
        private const uint BOUNCE_CONTINUE_DIELECTRIC = 2;

        private const int MaxMirrorBounces = 2;
        private const int MaxRefractionBounces = 2;
        // Hard safety cap on total Trace() calls per sample - the per-kind counters
        // above already bound the path length to at most MaxMirrorBounces +
        // MaxRefractionBounces continuations, so this is never the limiting factor in
        // practice.
        private const int MaxTotalBounces = 8;

        public unsafe static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var camera = launchParams.camera;

            LCG rng = new LCG((uint)(ix + (camera.width * iy)), (uint)launchParams.FrameID);

            int numPixelSamples = launchParams.NumPixelSamples;
            Vec3 pixelColor = new Vec3(0f, 0f, 0f);
            Vec3 pixelNormal = new Vec3(0f, 0f, 0f);
            Vec3 pixelAlbedo = new Vec3(0f, 0f, 0f);
            for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
            {
                float screenX = (2f * ((ix + rng.Next()) * camera.reciprocalWidth)) - 1f;
                float screenY = (2f * ((iy + rng.Next()) * camera.reciprocalHeight)) - 1f;

                Vec3 rayOrigin = camera.origin;
                Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                    new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

                Vec3 throughput = new Vec3(1f, 1f, 1f);
                Vec3 sampleRadiance = new Vec3(0f, 0f, 0f);
                Vec3 sampleNormal = new Vec3(0f, 0f, 0f);
                Vec3 sampleAlbedo = new Vec3(0f, 0f, 0f);
                int mirrorBounces = 0;
                int refractionBounces = 0;

                for (int bounce = 0; bounce < MaxTotalBounces; bounce++)
                {
                    uint p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0;
                    uint p13 = 0, p14 = 0, p15 = 0, p16 = 0, p17 = 0, p18 = 0;
                    OptixTrace.Trace(
                        launchParams.traversable,
                        (rayOrigin.x, rayOrigin.y, rayOrigin.z),
                        (rayDir.x, rayDir.y, rayDir.z),
                        1e-3f,
                        1e20f,
                        0.0f,
                        0xff,
                        OptixRayFlags.OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        RADIANCE_RAY_TYPE,
                        RAY_TYPE_COUNT,
                        RADIANCE_RAY_TYPE,
                        ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8, ref p9, ref p10, ref p11, ref p12,
                        ref p13, ref p14, ref p15, ref p16, ref p17, ref p18);

                    sampleRadiance += throughput * new Vec3(Interop.IntAsFloat(p0), Interop.IntAsFloat(p1), Interop.IntAsFloat(p2));

                    // AOV guide buffers only ever reflect the primary ray's own hit
                    // (bounce 0), matching Sample11/12's convention - a mirror/glass
                    // primary hit still contributes its own normal/tint here, which is
                    // exactly what the denoiser needs to recognize that surface.
                    if (bounce == 0)
                    {
                        sampleNormal = new Vec3(Interop.IntAsFloat(p13), Interop.IntAsFloat(p14), Interop.IntAsFloat(p15));
                        sampleAlbedo = new Vec3(Interop.IntAsFloat(p16), Interop.IntAsFloat(p17), Interop.IntAsFloat(p18));
                    }

                    uint flag = p3;
                    if (flag == BOUNCE_TERMINAL)
                        break;

                    if (flag == BOUNCE_CONTINUE_MIRROR)
                    {
                        mirrorBounces++;
                        if (mirrorBounces > MaxMirrorBounces)
                            break;
                    }
                    else
                    {
                        refractionBounces++;
                        if (refractionBounces > MaxRefractionBounces)
                            break;
                    }

                    throughput *= new Vec3(Interop.IntAsFloat(p10), Interop.IntAsFloat(p11), Interop.IntAsFloat(p12));
                    rayOrigin = new Vec3(Interop.IntAsFloat(p4), Interop.IntAsFloat(p5), Interop.IntAsFloat(p6));
                    rayDir = new Vec3(Interop.IntAsFloat(p7), Interop.IntAsFloat(p8), Interop.IntAsFloat(p9));

                    if (throughput.lengthSquared() < 1e-6f)
                        break;
                }

                pixelColor += sampleRadiance;
                pixelNormal += sampleNormal;
                pixelAlbedo += sampleAlbedo;
            }
            pixelColor /= (float)numPixelSamples;
            pixelNormal /= (float)numPixelSamples;
            pixelAlbedo /= (float)numPixelSamples;

            long fbIndex = ix + (iy * camera.width);
            if (launchParams.FrameID > 0)
            {
                Vec4 previous = launchParams.ColorBuffer[fbIndex];
                float weight = launchParams.FrameID;
                pixelColor = new Vec3(
                    ((weight * previous.x) + pixelColor.x) / (weight + 1f),
                    ((weight * previous.y) + pixelColor.y) / (weight + 1f),
                    ((weight * previous.z) + pixelColor.z) / (weight + 1f));
            }

            launchParams.ColorBuffer[fbIndex] = new Vec4(pixelColor.x, pixelColor.y, pixelColor.z, 1f);

            // Unlike ColorBuffer, AlbedoBuffer/NormalBuffer are NOT blended across
            // frames - each frame overwrites them fresh, matching Sample12's
            // devicePrograms.cs (these are the denoiser's AOV guide inputs, not part of
            // the progressively-accumulated image).
            launchParams.NormalBuffer[fbIndex] = new Vec4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1f);
            launchParams.AlbedoBuffer[fbIndex] = new Vec4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1f);
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            float t = 0.5f * (dy + 1f);
            Vec3 sky = launchParams.BackgroundBottom + (t * (launchParams.BackgroundTop - launchParams.BackgroundBottom));
            SetTerminalPayload(sky, new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f));
        }

        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            uint hitKind = OptixGetHitKind.Value;

            // Volume-grid hits are handled entirely separately: material is
            // data-dependent on which voxel the intersection program's DDA loop found,
            // not on primitive index (there's only one GAS primitive for the whole
            // grid), so material lookup bypasses the per-primitive SBT convention every
            // other hit kind uses - see ShadeVolumeGrid.
            if (hitKind >= IntersectionPrograms.HitKindVolumeGridFaceBase && hitKind < IntersectionPrograms.HitKindVolumeGridFaceBase + 6)
            {
                ShadeVolumeGrid(launchParams, rayDir, hitKind - IntersectionPrograms.HitKindVolumeGridFaceBase);
                return;
            }

            // Geometry is recovered analytically here rather than threaded through
            // intersection-program attributes (docs/SAMPLE13_PLAN.md design (b)) - a
            // built-in triangle hit (hitKind >= TriangleFrontFace) interpolates from the
            // triangle buffers as before; any other hitKind is a custom primitive
            // recomputed from its own per-primitive parameter buffer plus the hit
            // distance/ray.
            Vec3 surfPos;
            Vec3 rawGeometricNormal;
            Vec3 shadingNormal;
            Vec2 uv = new Vec2(0f, 0f); // custom primitives have no UV parameterization in this sample

            if (hitKind >= OptixGetHitKind.TriangleFrontFace)
            {
                uint primId = OptixGetPrimitiveIndex.Value;
                Vec3i tri = launchParams.Indices[primId];
                var (bu, bv) = OptixGetTriangleBarycentrics.Value;
                float bw = 1f - bu - bv;

                Vec3 a = launchParams.Vertices[tri.x];
                Vec3 b = launchParams.Vertices[tri.y];
                Vec3 c = launchParams.Vertices[tri.z];
                rawGeometricNormal = Vec3.unitVector(Vec3.cross(b - a, c - a));

                Vec3 n0 = launchParams.Normals[tri.x];
                Vec3 n1 = launchParams.Normals[tri.y];
                Vec3 n2 = launchParams.Normals[tri.z];
                shadingNormal = Vec3.unitVector((bw * n0) + (bu * n1) + (bv * n2));

                surfPos = (bw * a) + (bu * b) + (bv * c);

                Vec2 t0 = launchParams.TexCoords[tri.x];
                Vec2 t1 = launchParams.TexCoords[tri.y];
                Vec2 t2 = launchParams.TexCoords[tri.z];
                uv = new Vec2((bw * t0.x) + (bu * t1.x) + (bv * t2.x), (bw * t0.y) + (bu * t1.y) + (bv * t2.y));
            }
            else
            {
                // Custom primitive - hit position reconstructed from optixGetRayTmax(),
                // which inside closest-hit is the parametric distance of the accepted
                // hit; normal recomputed analytically per primitive kind (hitKind).
                var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
                Vec3 rayOrigin = new Vec3(ox, oy, oz);
                float hitT = OptixGetRayTmax.Value;

                surfPos = rayOrigin + (hitT * rayDir);
                rawGeometricNormal = ComputeCustomPrimitiveNormal(launchParams, hitKind, surfPos);
                shadingNormal = rawGeometricNormal;
            }

            // outwardNormal always faces against the incoming ray (frontFace == true
            // means the ray hit the side the raw cross-product normal already faces) -
            // needed as-is (not just for shading) by the dielectric branch to determine
            // which medium the ray is entering/leaving.
            bool frontFace = Vec3.dot(rayDir, rawGeometricNormal) < 0f;
            Vec3 outwardNormal = frontFace ? rawGeometricNormal : -rawGeometricNormal;

            if (Vec3.dot(outwardNormal, shadingNormal) < 0f)
                shadingNormal -= 2f * Vec3.dot(outwardNormal, shadingNormal) * outwardNormal;
            shadingNormal = Vec3.unitVector(shadingNormal);

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;

            // Shading dispatch order mirrors the reference's Material field-value
            // dispatch exactly: Transparency > 0 -> dielectric; else Reflectivity >= 0.9
            // -> mirror; else -> diffuse (Oren-Nayar + ambient).
            if (sbtData->Transparency > 0f)
            {
                ShadeDielectric(launchParams, sbtData, rayDir, outwardNormal, frontFace, surfPos);
                return;
            }

            if (sbtData->Reflectivity >= 0.9f)
            {
                Vec3 mirrorAlbedo = SampleAlbedo(sbtData, surfPos, uv);
                ShadeMirror(sbtData, mirrorAlbedo, rayDir, shadingNormal, outwardNormal, surfPos);
                return;
            }

            Vec3 albedo = SampleAlbedo(sbtData, surfPos, uv);

            ShadeDiffuse(launchParams, albedo, sbtData->Emission, surfPos, shadingNormal, outwardNormal, rayDir);
        }

        // Oren-Nayar + ambient + multi-light shading, shared between the ordinary
        // diffuse dispatch branch above and the volume grid's per-voxel material lookup
        // (ShadeVolumeGrid) - identical lighting math, only how the material/albedo was
        // obtained differs between the two callers.
        private unsafe static void ShadeDiffuse(LaunchParams launchParams, Vec3 albedo, Vec3 emission, Vec3 surfPos, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 rayDir)
        {
            Vec3 pixelColor = emission + (launchParams.AmbientColor * launchParams.AmbientIntensity * albedo);
            Vec3 viewDir = Vec3.unitVector(-rayDir);

            for (int i = 0; i < launchParams.NumPointLights; i++)
            {
                PointLightGpu light = launchParams.PointLights[i];
                Vec3 toLight = light.Position - surfPos;
                float dist2 = toLight.lengthSquared();
                float dist = XMath.Sqrt(dist2);
                Vec3 lightDir = toLight / dist;

                float NdotL = Vec3.dot(shadingNormal, lightDir);
                if (NdotL <= 0f)
                    continue;

                Vec3 transmittance = ShadowTransmittance(launchParams, surfPos, outwardNormal, lightDir, dist);
                if (transmittance.lengthSquared() <= 0f)
                    continue;

                float atten = light.Intensity / dist2;
                Vec3 bsdf = OrenNayarBRDF(albedo, shadingNormal, viewDir, lightDir, DiffuseSigmaDeg);
                pixelColor += (atten * NdotL) * bsdf * light.Color * transmittance;
            }

            SetTerminalPayload(pixelColor, shadingNormal, albedo);
        }

        // The volume grid is a single GAS primitive whose per-voxel material can't be
        // expressed via OptiX's per-primitive SbtIndexOffsetBuffer (that's keyed by
        // primitive index, not by data read during intersection) - so material lookup
        // goes directly through LaunchParams.Materials/VoxelMaterialIds instead of
        // OptixGetSbtDataPointer, a deliberate deviation from every other primitive kind
        // in this sample (see docs/SAMPLE13_PLAN.md's note on this and the comment on
        // LaunchParams.Materials).
        private unsafe static void ShadeVolumeGrid(LaunchParams launchParams, Vec3 rayDir, uint faceCode)
        {
            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            Vec3 rayOrigin = new Vec3(ox, oy, oz);
            float hitT = OptixGetRayTmax.Value;
            Vec3 surfPos = rayOrigin + (hitT * rayDir);

            Vec3 rawGeometricNormal = VolumeGridFaceNormal(faceCode);
            bool frontFace = Vec3.dot(rayDir, rawGeometricNormal) < 0f;
            Vec3 outwardNormal = frontFace ? rawGeometricNormal : -rawGeometricNormal;

            Vec3i dims = launchParams.VolumeDims;
            Vec3 local = surfPos - launchParams.VolumeGridMin;
            int ix = XMath.Clamp((int)XMath.Floor(local.x / launchParams.VolumeVoxelSize.x), 0, dims.x - 1);
            int iy = XMath.Clamp((int)XMath.Floor(local.y / launchParams.VolumeVoxelSize.y), 0, dims.y - 1);
            int iz = XMath.Clamp((int)XMath.Floor(local.z / launchParams.VolumeVoxelSize.z), 0, dims.z - 1);
            uint voxel = launchParams.VoxelMaterialIds[(ix * dims.y * dims.z) + (iy * dims.z) + iz];
            // voxel == 0 (empty) never reaches here - the intersection program's DDA
            // only reports a hit on a non-empty voxel.
            MaterialSbtData mat = launchParams.Materials[(int)voxel - 1];

            Vec3 albedo = mat.MaterialKind == MaterialSbtData.Checker ? SampleChecker(mat, surfPos) : mat.Albedo;
            ShadeDiffuse(launchParams, albedo, mat.Emission, surfPos, outwardNormal, outwardNormal, rayDir);
        }

        private static Vec3 VolumeGridFaceNormal(uint faceCode)
        {
            switch (faceCode)
            {
                case 0: return new Vec3(1f, 0f, 0f);
                case 1: return new Vec3(-1f, 0f, 0f);
                case 2: return new Vec3(0f, 1f, 0f);
                case 3: return new Vec3(0f, -1f, 0f);
                case 4: return new Vec3(0f, 0f, 1f);
                default: return new Vec3(0f, 0f, -1f);
            }
        }

        // Analytic normal recomputation per custom-primitive kind (docs/SAMPLE13_PLAN.md
        // design (b)) - avoids threading attributes through OptixReportIntersection.
        private unsafe static Vec3 ComputeCustomPrimitiveNormal(LaunchParams launchParams, uint hitKind, Vec3 surfPos)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            switch (hitKind)
            {
                case IntersectionPrograms.HitKindSphere:
                {
                    SphereData sphere = launchParams.Spheres[primId];
                    return Vec3.unitVector((surfPos - sphere.Center) / sphere.Radius);
                }
                case IntersectionPrograms.HitKindBox:
                {
                    BoxData box = launchParams.Boxes[primId];
                    Vec3 center = (box.Min + box.Max) * 0.5f;
                    Vec3 halfExtents = (box.Max - box.Min) * 0.5f;
                    Vec3 pc = surfPos - center;
                    float dx = XMath.Abs(pc.x) - halfExtents.x;
                    float dy = XMath.Abs(pc.y) - halfExtents.y;
                    float dz = XMath.Abs(pc.z) - halfExtents.z;
                    if (dx > dy && dx > dz)
                        return new Vec3(pc.x >= 0f ? 1f : -1f, 0f, 0f);
                    if (dy > dz)
                        return new Vec3(0f, pc.y >= 0f ? 1f : -1f, 0f);
                    return new Vec3(0f, 0f, pc.z >= 0f ? 1f : -1f);
                }
                case IntersectionPrograms.HitKindCylinderY:
                {
                    CylinderYData cyl = launchParams.CylindersY[primId];
                    const float capEpsilon = 1e-3f;
                    if (XMath.Abs(surfPos.y - cyl.YMax) < capEpsilon)
                        return new Vec3(0f, 1f, 0f);
                    if (XMath.Abs(surfPos.y - cyl.YMin) < capEpsilon)
                        return new Vec3(0f, -1f, 0f);
                    return Vec3.unitVector(new Vec3(surfPos.x - cyl.Center.x, 0f, surfPos.z - cyl.Center.z));
                }
                case IntersectionPrograms.HitKindDisk:
                {
                    DiskData disk = launchParams.Disks[primId];
                    return disk.Normal;
                }
                case IntersectionPrograms.HitKindXYRect:
                    return new Vec3(0f, 0f, 1f);
                case IntersectionPrograms.HitKindXZRect:
                    return new Vec3(0f, 1f, 0f);
                default: // HitKindYZRect
                    return new Vec3(1f, 0f, 0f);
            }
        }

        private unsafe static void ShadeMirror(MaterialSbtData* sbtData, Vec3 albedo, Vec3 rayDir, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 surfPos)
        {
            Vec3 reflectDir = Vec3.reflect(shadingNormal, rayDir);
            Vec3 newOrigin = surfPos + (1e-3f * outwardNormal);
            SetContinuePayload(sbtData->Emission, BOUNCE_CONTINUE_MIRROR, newOrigin, reflectDir, albedo, shadingNormal, albedo);
        }

        // Fresnel-weighted stochastic reflect-XOR-refract pick (not the reference's
        // branch-both-paths-at-once) - converges to the same result over many
        // accumulated frames given the existing progressive accumulation. Total
        // internal reflection forces the reflect branch unconditionally (no valid
        // refraction direction exists). A fractional Transparency isn't currently
        // exercised by any Sample13 scene (materials use 1.0, matching the reference's
        // actual glass materials) - if a future scene wants partial transparency,
        // revisit how it should modulate this probability split.
        private unsafe static void ShadeDielectric(LaunchParams launchParams, MaterialSbtData* sbtData, Vec3 rayDir, Vec3 outwardNormal, bool frontFace, Vec3 surfPos)
        {
            float etaFrom = frontFace ? 1f : sbtData->IndexOfRefraction;
            float etaTo = frontFace ? sbtData->IndexOfRefraction : 1f;
            float eta = etaFrom / etaTo;

            bool didRefract = TryRefract(rayDir, outwardNormal, eta, out Vec3 refractDir);

            float cosTheta = XMath.Clamp(Vec3.dot(-rayDir, outwardNormal), 0f, 1f);
            float fresnel = didRefract ? FresnelSchlick(cosTheta, etaFrom, etaTo) : 1f;
            fresnel = XMath.Clamp(fresnel + (sbtData->Reflectivity * (1f - fresnel)), 0f, 1f);

            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            uint primId = OptixGetPrimitiveIndex.Value;
            LCG rng = new LCG((uint)(ix + (1000003 * iy)), (uint)((launchParams.FrameID * 7919) + primId));

            bool chooseReflect = !didRefract || rng.Next() < fresnel;

            Vec3 newDir;
            Vec3 tint;
            Vec3 offsetNormal;
            if (chooseReflect)
            {
                newDir = Vec3.reflect(outwardNormal, rayDir);
                tint = new Vec3(1f, 1f, 1f);
                offsetNormal = outwardNormal;
            }
            else
            {
                newDir = refractDir;
                tint = sbtData->TransmissionColor;
                offsetNormal = -outwardNormal;
            }

            Vec3 newOrigin = surfPos + (1e-3f * offsetNormal);
            SetContinuePayload(sbtData->Emission, BOUNCE_CONTINUE_DIELECTRIC, newOrigin, newDir, tint, outwardNormal, tint);
        }

        // v = incident ray direction (pointing into the surface), n = outward normal
        // (facing against v, same side as the ray origin), niOverNt = etaFrom/etaTo -
        // matches Vectors.cs's existing Vec3.refract convention (Shirley's "Ray Tracing
        // in One Weekend" formula), reimplemented here with an explicit success flag
        // rather than relying on that helper's silent "return v unchanged" TIR fallback.
        private static bool TryRefract(Vec3 v, Vec3 n, float niOverNt, out Vec3 refracted)
        {
            Vec3 uv = Vec3.unitVector(v);
            float dt = Vec3.dot(uv, n);
            float discriminant = 1f - (niOverNt * niOverNt * (1f - (dt * dt)));
            if (discriminant > 0f)
            {
                refracted = (niOverNt * (uv - (n * dt))) - (n * XMath.Sqrt(discriminant));
                return true;
            }
            refracted = Vec3.reflect(n, v);
            return false;
        }

        private static float FresnelSchlick(float cosTheta, float etaFrom, float etaTo)
        {
            float r0 = (etaFrom - etaTo) / (etaFrom + etaTo);
            r0 *= r0;
            float x = 1f - cosTheta;
            float x2 = x * x;
            return r0 + ((1f - r0) * x2 * x2 * x);
        }

        // Multi-hit colored transmittance shadow ray (docs/SAMPLE13_PLAN.md design
        // decision (f)) - walks through transparent occluders via __anyhit__shadow's
        // optixIgnoreIntersection, accumulating TransmissionColor*Transparency per hit
        // up to MaxRefractions, unlike a plain binary occlusion test.
        private unsafe static Vec3 ShadowTransmittance(LaunchParams launchParams, Vec3 surfPos, Vec3 outwardNormal, Vec3 lightDir, float lightDist)
        {
            uint t0 = Interop.FloatAsInt(1f);
            uint t1 = Interop.FloatAsInt(1f);
            uint t2 = Interop.FloatAsInt(1f);
            uint hitCount = 0;

            OptixTrace.Trace(
                launchParams.traversable,
                (surfPos.x + (1e-3f * outwardNormal.x), surfPos.y + (1e-3f * outwardNormal.y), surfPos.z + (1e-3f * outwardNormal.z)),
                (lightDir.x, lightDir.y, lightDir.z),
                1e-3f,
                lightDist * (1f - 1e-3f),
                0.0f,
                0xff,
                OptixRayFlags.OPTIX_RAY_FLAG_NONE,
                SHADOW_RAY_TYPE,
                RAY_TYPE_COUNT,
                SHADOW_RAY_TYPE,
                ref t0, ref t1, ref t2, ref hitCount);

            return new Vec3(Interop.IntAsFloat(t0), Interop.IntAsFloat(t1), Interop.IntAsFloat(t2));
        }

        // Direct port of the reference's OrenNayarBRDF (RayTracing/RaytraceRenderer.cs) -
        // NdotL is intentionally NOT baked in here (applied by the caller, matching
        // "radiance += beta * bsdf*nDotL*Li*transmittance" in the reference). Guards
        // against normalizing a near-zero perpendicular-component vector (degenerates to
        // NaN via 0/0 when L or V is nearly parallel to N) by falling back to
        // cosPhiDiff = 0, which only ever discards the (already small/vanishing) azimuthal
        // correction term.
        private static Vec3 OrenNayarBRDF(Vec3 albedo, Vec3 N, Vec3 V, Vec3 L, float sigmaDeg)
        {
            float sigma = sigmaDeg * (XMath.PI / 180f);
            float sigma2 = sigma * sigma;
            float A = 1f - (sigma2 / (2f * (sigma2 + 0.33f)));
            float B = 0.45f * sigma2 / (sigma2 + 0.09f);

            float NdotL = XMath.Clamp(Vec3.dot(N, L), 0f, 1f);
            float NdotV = XMath.Clamp(Vec3.dot(N, V), 0f, 1f);
            float thetaI = XMath.Acos(NdotL);
            float thetaR = XMath.Acos(NdotV);
            float alpha = XMath.Max(thetaI, thetaR);
            float beta = XMath.Min(thetaI, thetaR);

            Vec3 lightPerpRaw = L - (N * NdotL);
            Vec3 viewPerpRaw = V - (N * NdotV);
            float lightPerpLenSq = lightPerpRaw.lengthSquared();
            float viewPerpLenSq = viewPerpRaw.lengthSquared();

            float cosPhiDiff = 0f;
            if (lightPerpLenSq > 1e-8f && viewPerpLenSq > 1e-8f)
            {
                Vec3 lightPerp = lightPerpRaw / XMath.Sqrt(lightPerpLenSq);
                Vec3 viewPerp = viewPerpRaw / XMath.Sqrt(viewPerpLenSq);
                cosPhiDiff = XMath.Clamp(Vec3.dot(lightPerp, viewPerp), -1f, 1f);
            }

            float factor = (A + (B * cosPhiDiff * XMath.Sin(alpha) * XMath.Tan(beta))) * (1f / XMath.PI);
            return albedo * factor;
        }

        // Direct port of the reference's Checker closure (position-evaluated instead of a
        // captured Func<>) - floor(pos.X/scale) + floor(pos.Z/scale) parity picks between
        // the two flat colors.
        private static Vec3 SampleChecker(MaterialSbtData mat, Vec3 pos)
        {
            int cx = (int)XMath.Floor(pos.x / mat.CheckerScale);
            int cz = (int)XMath.Floor(pos.z / mat.CheckerScale);
            bool even = ((cx + cz) & 1) == 0;
            return even ? mat.Albedo : mat.CheckerColorB;
        }

        // Direct port of the reference's SampleAlbedo (RaytraceRenderer.cs) -
        // TextureObject == 0 means no texture (matches Sample08's convention). UV is
        // only ever non-zero for triangle hits (custom primitives have no UV
        // parameterization in this sample), so a textured custom primitive is not
        // currently possible - not needed by any Sample13 scene.
        private unsafe static Vec3 SampleAlbedo(MaterialSbtData* sbtData, Vec3 surfPos, Vec2 uv)
        {
            Vec3 baseAlbedo = sbtData->MaterialKind == MaterialSbtData.Checker
                ? SampleChecker(*sbtData, surfPos)
                : sbtData->Albedo;

            if (sbtData->TextureObject == 0)
                return baseAlbedo;

            var (tr, tg, tb, _) = CudaTex2D.Sample(sbtData->TextureObject, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            Vec3 texSample = new Vec3(tr, tg, tb);
            return (baseAlbedo * (1f - sbtData->TextureWeight)) + (texSample * sbtData->TextureWeight);
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        public static void __closesthit__shadow(LaunchParams launchParams)
        { }

        // Accumulates colored transmittance through transparent occluders instead of a
        // plain binary occlusion test (docs/SAMPLE13_PLAN.md design (f)). Opaque hits are
        // a no-op here, which leaves OptiX's default any-hit behavior in effect: accept
        // the hit and terminate the ray, i.e. fully block the light - exactly matching
        // the reference's "opaque fully blocks" rule.
        public unsafe static void __anyhit__shadow(LaunchParams launchParams)
        {
            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            if (sbtData->Transparency <= 0f)
                return;

            const uint maxRefractions = 2;
            const float transmittanceEpsilon = 1e-6f;

            Vec3 transmittance = new Vec3(
                Interop.IntAsFloat(OptixPayload.Payload0),
                Interop.IntAsFloat(OptixPayload.Payload1),
                Interop.IntAsFloat(OptixPayload.Payload2));
            uint hitCount = OptixPayload.Payload3 + 1;

            transmittance *= sbtData->TransmissionColor * sbtData->Transparency;

            OptixPayload.Payload0 = Interop.FloatAsInt(transmittance.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(transmittance.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(transmittance.z);
            OptixPayload.Payload3 = hitCount;

            if (hitCount < maxRefractions && transmittance.lengthSquared() > transmittanceEpsilon)
                OptixIgnoreIntersection.Ignore();
            // else: let this hit stick (accept & terminate) - the accumulated
            // transmittance above is the final answer, matching the reference's
            // early-out-on-negligible-transmittance / MaxRefractions-exceeded behavior.
        }

        public static void __miss__shadow(LaunchParams launchParams)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(1f);
            OptixPayload.Payload1 = Interop.FloatAsInt(1f);
            OptixPayload.Payload2 = Interop.FloatAsInt(1f);
        }

        // normal/albedo (payloads 13-18) are only ever read back by raygen from the
        // bounce==0 Trace() call (see __raygen__renderFrame) - matching Sample11/12's
        // AOV guide-buffer convention of reflecting only the primary-ray hit, not
        // anything accumulated across bounces - but every shading branch below still
        // populates them on every call, since raygen has no way to know in advance
        // which bounce a given closest-hit invocation corresponds to.
        private static void SetTerminalPayload(Vec3 contribution, Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(contribution.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(contribution.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(contribution.z);
            OptixPayload.Payload3 = BOUNCE_TERMINAL;
            SetAovPayload(normal, albedo);
        }

        private static void SetContinuePayload(Vec3 contribution, uint flag, Vec3 newOrigin, Vec3 newDir, Vec3 tint, Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(contribution.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(contribution.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(contribution.z);
            OptixPayload.Payload3 = flag;
            OptixPayload.Payload4 = Interop.FloatAsInt(newOrigin.x);
            OptixPayload.Payload5 = Interop.FloatAsInt(newOrigin.y);
            OptixPayload.Payload6 = Interop.FloatAsInt(newOrigin.z);
            OptixPayload.Payload7 = Interop.FloatAsInt(newDir.x);
            OptixPayload.Payload8 = Interop.FloatAsInt(newDir.y);
            OptixPayload.Payload9 = Interop.FloatAsInt(newDir.z);
            OptixPayload.Payload10 = Interop.FloatAsInt(tint.x);
            OptixPayload.Payload11 = Interop.FloatAsInt(tint.y);
            OptixPayload.Payload12 = Interop.FloatAsInt(tint.z);
            SetAovPayload(normal, albedo);
        }

        private static void SetAovPayload(Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload13 = Interop.FloatAsInt(normal.x);
            OptixPayload.Payload14 = Interop.FloatAsInt(normal.y);
            OptixPayload.Payload15 = Interop.FloatAsInt(normal.z);
            OptixPayload.Payload16 = Interop.FloatAsInt(albedo.x);
            OptixPayload.Payload17 = Interop.FloatAsInt(albedo.y);
            OptixPayload.Payload18 = Interop.FloatAsInt(albedo.z);
        }

        // Gamma-corrected (sqrt) tonemap + row flip for display, matching Sample12's.
        public static void tonemapAndFlip(Index1D index, int width, int height, ArrayView<Vec4> hdr, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);
            int newIndex = ((y * width) + x) * 4;

            Vec4 color = hdr[index];
            byte r = (byte)(XMath.Clamp(XMath.Sqrt(color.x), 0f, 1f) * 255f);
            byte g = (byte)(XMath.Clamp(XMath.Sqrt(color.y), 0f, 1f) * 255f);
            byte b = (byte)(XMath.Clamp(XMath.Sqrt(color.z), 0f, 1f) * 255f);

            dest[newIndex] = b;
            dest[newIndex + 1] = g;
            dest[newIndex + 2] = r;
            dest[newIndex + 3] = 0xff;
        }
    }
}
