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

using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

namespace Sample19
{
    public static class Program
    {
        public static void Main()
        {
            var gameWindowSettings = GameWindowSettings.Default;
            var nativeWindowSettings = new NativeWindowSettings
            {
                ClientSize = new Vector2i(800, 600),
                Title = "Sample19 - Denoiser Temporal/AOV Guide Layers",
                Flags = ContextFlags.ForwardCompatible,
                Profile = ContextProfile.Core,
                APIVersion = new System.Version(3, 3),
            };

            using var window = new RenderWindow(gameWindowSettings, nativeWindowSettings);
            window.Run();
        }
    }
}
