// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: App.xaml.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX;
using System;
using System.Windows;
using System.Windows.Threading;

namespace Sample11
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            AppDomain.CurrentDomain.UnhandledException += (s, e) =>
                PrintException(e.ExceptionObject as Exception);
            DispatcherUnhandledException += (s, e) =>
            {
                PrintException(e.Exception);
                e.Handled = true;
                Environment.Exit(1);
            };
        }

        private static void PrintException(Exception ex)
        {
            Console.Error.WriteLine("Unhandled exception:");
            for (var current = ex; current != null; current = current.InnerException)
            {
                Console.Error.WriteLine($"--- {current.GetType().FullName}: {current.Message}");
                if (current is OptixException optixEx)
                    Console.Error.WriteLine($"    OptixResult: {optixEx.OptixResult}");
                Console.Error.WriteLine(current.StackTrace);
            }
        }
    }
}
