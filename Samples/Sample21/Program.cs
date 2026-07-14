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

namespace Sample21
{
    public static class Program
    {
        public static void Main()
        {
            var gameWindowSettings = GameWindowSettings.Default;
            var nativeWindowSettings = new NativeWindowSettings
            {
                ClientSize = new Vector2i(1024, 768),
                Title = "Sample21 - Opacity Micromaps (Cutout Cards)",
                Flags = ContextFlags.ForwardCompatible,
                Profile = ContextProfile.Core,
                APIVersion = new System.Version(3, 3),
            };

            using var window = new RenderWindow(gameWindowSettings, nativeWindowSettings);
            window.Run();
        }
    }
}
