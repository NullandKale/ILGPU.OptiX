using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Sample06
{
    public struct Camera
    {
        public SpecializedValue<int> height { get; set; }
        public SpecializedValue<int> width { get; set; }

        public Vec3 noHitColor { get; set; }

        public float verticalFov { get; set; }
        public float worldScale { get; set; }

        public Vec3 origin { get; set; }
        public Vec3 lookAt { get; set; }
        public Vec3 up { get; set; }
        public OrthoNormalBasis axis { get; set; }

        public float aspectRatio { get; set; }
        public float cameraPlaneDist { get; set; }
        public float reciprocalHeight { get; set; }
        public float reciprocalWidth { get; set; }


        public Camera(Camera camera, Vec3 movement, Vec3 turn)
        {
            this.width = camera.width;
            this.height = camera.height;
            this.noHitColor = camera.noHitColor;
            this.worldScale = camera.worldScale;

            Vector4 temp = camera.lookAt - camera.origin;

            if (turn.y != 0)
            {
                temp += Vector4.Transform(temp, Matrix4x4.CreateFromAxisAngle(Vec3.cross(Vec3.cross(camera.up, (camera.lookAt - camera.origin)), (camera.lookAt - camera.origin)), (float)turn.y));
            }
            if (turn.x != 0)
            {
                temp += Vector4.Transform(temp, Matrix4x4.CreateFromAxisAngle(Vec3.cross(camera.up, (camera.lookAt - camera.origin)), (float)turn.x));
            }

            lookAt = camera.origin + Vec3.unitVector(temp);

            this.origin = camera.origin + movement;
            this.lookAt += movement;
            this.up = camera.up;

            axis = OrthoNormalBasis.fromZY(Vec3.unitVector(lookAt - origin), up);

            aspectRatio = (width / (float)height);
            cameraPlaneDist = 1.0f / XMath.Tan(camera.verticalFov * XMath.PI / 360.0f);
            this.verticalFov = camera.verticalFov;
            reciprocalHeight = 1.0f / height;
            reciprocalWidth = 1.0f / width;
        }

        public Camera(Camera camera, int width, int height)
        {
            this.width = new SpecializedValue<int>(width);
            this.height = new SpecializedValue<int>(height);
            this.noHitColor = camera.noHitColor;
            this.worldScale = camera.worldScale;

            this.verticalFov = camera.verticalFov;

            this.origin = camera.origin;
            this.lookAt = camera.lookAt;
            this.up = camera.up;

            axis = OrthoNormalBasis.fromZY(Vec3.unitVector(lookAt - origin), up);

            aspectRatio = (width / (float)height);
            cameraPlaneDist = 1.0f / XMath.Tan(verticalFov * XMath.PI / 360.0f);
            reciprocalHeight = 1.0f / height;
            reciprocalWidth = 1.0f / width;
        }

        public Camera(Vec3 origin, Vec3 lookAt, Vec3 up, int width, int height, Vec3 noHitColor, float verticalFov, float worldScale)
        {
            this.width = new SpecializedValue<int>(width);
            this.height = new SpecializedValue<int>(height);
            this.noHitColor = noHitColor;
            this.worldScale = worldScale;

            this.verticalFov = verticalFov;
            this.origin = origin;
            this.lookAt = lookAt;
            this.up = up;

            axis = OrthoNormalBasis.fromZY(Vec3.unitVector(lookAt - origin), up);

            aspectRatio = (width / (float)height);
            cameraPlaneDist = 1.0f / XMath.Tan(verticalFov * XMath.PI / 360.0f);
            reciprocalHeight = 1.0f / height;
            reciprocalWidth = 1.0f / width;
        }
    }

    // Mouse-driven orbit/pan/dolly camera controls, ported from the behavior of the
    // optix7course C++ reference's InspectModeManip ("inspect mode": left-drag orbits
    // around a fixed look-at point, right-drag dollies toward/away from it, middle-drag
    // pans). Deliberately separate from the existing (dead, unused, buggy-approximation)
    // Camera(Camera, Vec3 movement, Vec3 turn) constructor, which implements a different
    // "fly mode" rotate-in-place behavior and is left untouched.
    public static class CameraMotion
    {
        private const float DegreesPerDragFraction = 150f;
        private const float PixelsPerMove = 10f;
        private const float MinDistance = 0.1f;
        private const float MinPolarDegrees = 2f;

        public static Camera Orbit(Camera camera, float dxFraction, float dyFraction)
        {
            float yaw = -dxFraction * DegreesPerDragFraction * XMath.PI / 180f;
            float pitch = -dyFraction * DegreesPerDragFraction * XMath.PI / 180f;

            Vec3 lookAt = camera.lookAt;
            Vec3 worldUp = camera.up;
            Vec3 offset = camera.origin - lookAt;

            // yaw around world up
            offset = Vector4.Transform((Vector4)offset, Matrix4x4.CreateFromAxisAngle(worldUp, yaw));

            // pitch around the current local right axis
            Vec3 forward = Vec3.unitVector(-offset);
            Vec3 right = Vec3.unitVector(Vec3.cross(worldUp, forward));
            Vec3 pitchedOffset = Vector4.Transform((Vector4)offset, Matrix4x4.CreateFromAxisAngle(right, pitch));

            // reject the pitch step alone (keep yaw) if it would cross too close to the poles
            float minPolarRad = MinPolarDegrees * XMath.PI / 180f;
            Vec3 newForward = Vec3.unitVector(-pitchedOffset);
            float polar = XMath.Acos(XMath.Clamp(Vec3.dot(newForward, worldUp), -1f, 1f));
            if (polar >= minPolarRad && polar <= XMath.PI - minPolarRad)
            {
                offset = pitchedOffset;
            }

            camera.origin = lookAt + offset;
            camera.axis = OrthoNormalBasis.fromZY(Vec3.unitVector(camera.lookAt - camera.origin), worldUp);
            return camera;
        }

        public static Camera Dolly(Camera camera, float dyFraction)
        {
            Vec3 offsetDir = Vec3.unitVector(camera.origin - camera.lookAt);
            float distance = Vec3.dist(camera.origin, camera.lookAt);
            float delta = dyFraction * PixelsPerMove * camera.worldScale;
            float newDistance = XMath.Max(distance - delta, MinDistance);

            camera.origin = camera.lookAt + (offsetDir * newDistance);
            camera.axis = OrthoNormalBasis.fromZY(Vec3.unitVector(camera.lookAt - camera.origin), camera.up);
            return camera;
        }

        public static Camera Pan(Camera camera, float dxFraction, float dyFraction)
        {
            float distance = Vec3.dist(camera.origin, camera.lookAt);
            Vec3 translation =
                ((-dxFraction * camera.axis.x) + (dyFraction * camera.axis.y))
                * PixelsPerMove * camera.worldScale * (distance / 100f);

            camera.origin += translation;
            camera.lookAt += translation;
            camera.axis = OrthoNormalBasis.fromZY(Vec3.unitVector(camera.lookAt - camera.origin), camera.up);
            return camera;
        }
    }

    public struct OrthoNormalBasis
    {
        public Vec3 x { get; set; }
        public Vec3 y { get; set; }
        public Vec3 z { get; set; }

        public OrthoNormalBasis(Vec3 x, Vec3 y, Vec3 z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }


        public Vec3 transform(Vec3 pos)
        {
            return x * pos.x + y * pos.y + z * pos.z;
        }


        public static OrthoNormalBasis fromXY(Vec3 x, Vec3 y)
        {
            Vec3 zz = Vec3.unitVector(Vec3.cross(x, y));
            Vec3 yy = Vec3.unitVector(Vec3.cross(zz, x));
            return new OrthoNormalBasis(x, yy, zz);
        }


        public static OrthoNormalBasis fromYX(Vec3 y, Vec3 x)
        {
            Vec3 zz = Vec3.unitVector(Vec3.cross(x, y));
            Vec3 xx = Vec3.unitVector(Vec3.cross(y, zz));
            return new OrthoNormalBasis(xx, y, zz);
        }


        public static OrthoNormalBasis fromXZ(Vec3 x, Vec3 z)
        {
            Vec3 yy = Vec3.unitVector(Vec3.cross(z, x));
            Vec3 zz = Vec3.unitVector(Vec3.cross(x, yy));
            return new OrthoNormalBasis(x, yy, zz);
        }


        public static OrthoNormalBasis fromZX(Vec3 z, Vec3 x)
        {
            Vec3 yy = Vec3.unitVector(Vec3.cross(z, x));
            Vec3 xx = Vec3.unitVector(Vec3.cross(yy, z));
            return new OrthoNormalBasis(xx, yy, z);
        }


        public static OrthoNormalBasis fromYZ(Vec3 y, Vec3 z)
        {
            Vec3 xx = Vec3.unitVector(Vec3.cross(y, z));
            Vec3 zz = Vec3.unitVector(Vec3.cross(xx, y));
            return new OrthoNormalBasis(xx, y, zz);
        }


        public static OrthoNormalBasis fromZY(Vec3 z, Vec3 y)
        {
            Vec3 xx = Vec3.unitVector(Vec3.cross(y, z));
            Vec3 yy = Vec3.unitVector(Vec3.cross(z, xx));
            return new OrthoNormalBasis(xx, yy, z);
        }


        public static OrthoNormalBasis fromZ(Vec3 z)
        {
            Vec3 xx;
            if (XMath.Abs(Vec3.dot(z, new Vec3(1, 0, 0))) > 0.99999f)
            {
                xx = Vec3.unitVector(Vec3.cross(new Vec3(0, 1, 0), z));
            }
            else
            {
                xx = Vec3.unitVector(Vec3.cross(new Vec3(1, 0, 0), z));
            }
            Vec3 yy = Vec3.unitVector(Vec3.cross(z, xx));
            return new OrthoNormalBasis(xx, yy, z);
        }
    }
}
