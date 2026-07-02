// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LCG.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample12
{
    // Port of optix7course's gdt::LCG<16> (common/gdt/gdt/random/random.h) - a 16-round
    // linear congruential generator seeded from two integers (here, pixel coordinates),
    // then advanced one step per call. State is a single uint, so it survives round-trips
    // through an OptiX payload register (see devicePrograms.cs) - unlike the C++
    // reference, there's no PRD pointer to carry it in device code here (ILGPU.OptiX
    // doesn't support pointer<->int payload packing), so the caller must thread it
    // through explicitly.
    public struct LCG
    {
        public uint State;

        public LCG(uint val0, uint val1)
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
            State = v0;
        }

        // Returns the next random value in [0, 1) and advances the state.
        public float Next()
        {
            const uint LCG_A = 1664525u;
            const uint LCG_C = 1013904223u;
            State = (LCG_A * State) + LCG_C;
            return (State & 0x00FFFFFFu) / (float)0x01000000u;
        }
    }
}
