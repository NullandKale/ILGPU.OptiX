// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Camera.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Sample14
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


        public static OrthoNormalBasis fromZY(Vec3 z, Vec3 y)
        {
            Vec3 xx = Vec3.unitVector(Vec3.cross(y, z));
            Vec3 yy = Vec3.unitVector(Vec3.cross(z, xx));
            return new OrthoNormalBasis(xx, yy, z);
        }
    }
}
