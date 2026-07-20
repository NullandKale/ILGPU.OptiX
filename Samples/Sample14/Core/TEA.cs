// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: TEA.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample14
{
    // Tiny Encryption Algorithm hash (16 rounds, matching optixIntro_04's tea<16> and
    // optix7course's gdt::LCG<16> seeding step) - a deterministic, well-mixed uint pair ->
    // uint hash used to seed an RNG stream from arbitrary integer keys (pixel coordinates,
    // frame index, bounce depth, ...). Extracted as its own function so any device code
    // needing a fresh, uncorrelated seed - not just LCG's own constructor - can call it
    // directly.
    internal static class TEA
    {
        internal static uint Hash(uint val0, uint val1)
        {
            uint v0 = val0;
            uint v1 = val1;
            uint s0 = 0;

            for (uint n = 0; n < 16; n++)
            {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }
            return v0;
        }
    }
}
