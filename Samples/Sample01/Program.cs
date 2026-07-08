// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Program.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.OptiX;
using ILGPU.Runtime.Cuda;
using System;

namespace Sample01
{
    // The simplest possible OptiX sample: create a CUDA accelerator and an OptiX
    // context, then exit. It renders nothing - this only demonstrates that the
    // CUDA/OptiX runtime initializes correctly on this machine.
    public class Program
    {
        static void Main()
        {
            try
            {
                Console.WriteLine("Initializing CUDA + OptiX...");
                using var context = Context.Create(b => b.Cuda());
                using var accelerator = context.CreateCudaAccelerator(0);
                Console.WriteLine($"Success: CUDA accelerator '{accelerator.Name}' and OptiX context initialized.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                Environment.Exit(1);
            }
        }
    }
}
