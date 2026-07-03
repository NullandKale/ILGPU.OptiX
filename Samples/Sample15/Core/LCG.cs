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

namespace Sample15
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

        // TEA-seeded (docs/SAMPLE15_PLAN.md Milestone M3, Core/TEA.cs) - the two integer
        // keys are hashed into a well-mixed initial state.
        public LCG(uint val0, uint val1)
        {
            State = TEA.Hash(val0, val1);
        }

        // Resumes a stream from a state carried across an OptiX payload round-trip
        // (Milestone M3's continuous raygen<->closesthit RNG stream), instead of
        // reseeding a fresh TEA hash - reseeding per shading call from (pixel, frame,
        // primitive) silently correlates rays that hit the same primitive on different
        // bounces, which this constructor exists to avoid.
        public LCG(uint state)
        {
            State = state;
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
