// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Tri.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample20
{
    /// <summary>Index triplet matching OptixIndicesFormat.UnsignedInt3's expected layout.</summary>
    public struct Tri
    {
        public uint A, B, C;

        public Tri(uint a, uint b, uint c)
        {
            A = a;
            B = b;
            C = c;
        }
    }
}
