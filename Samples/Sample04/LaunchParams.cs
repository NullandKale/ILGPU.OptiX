// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LaunchParams.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Device;

namespace Sample04
{
    public struct LaunchParams
    {
        public int FrameID;
        public OptixDeviceView<uint> ColorBuffer;
        public Camera camera;
        public ulong traversable;
    }
}
