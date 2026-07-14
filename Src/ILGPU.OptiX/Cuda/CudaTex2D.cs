// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: CudaTex2D.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.Cuda
{
    // Samples a bindless CUDA texture object (cudaTextureObject_t, created host-side by
    // CudaTextureObject) from device code, equivalent to CUDA's tex2D<float4>(). Not an
    // OptiX built-in - this is a raw PTX texture-fetch instruction, which needs
    // CudaAsm.EmitRef (rather than CudaAsm.Emit, which only supports a single output
    // parameter) since tex.2d.v4.f32.f32 writes all four color channels from one call.
    [CLSCompliant(false)]
    public static class CudaTex2D
    {
        public static (float R, float G, float B, float A) Sample(ulong textureObject, float u, float v)
        {
            Output<float> r = default;
            Output<float> g = default;
            Output<float> b = default;
            Output<float> a = default;
            Input<ulong> tex = textureObject;
            Input<float> uu = u;
            Input<float> vv = v;

            CudaAsm.EmitRef(
                "tex.2d.v4.f32.f32 {%0,%1,%2,%3}, [%4, {%5,%6}];",
                ref r, ref g, ref b, ref a, ref tex, ref uu, ref vv);

            return (r.Value, g.Value, b.Value, a.Value);
        }
    }
}
