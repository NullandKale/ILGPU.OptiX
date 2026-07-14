using ILGPU.OptiX;
using ILGPU.OptiX.Native;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace Sample05
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
