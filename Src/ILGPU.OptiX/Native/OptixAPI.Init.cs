// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAPI.Init.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Util;
using Microsoft.Win32;
using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Runtime.InteropServices;

// This binding targets NVIDIA OptiX SDK 9.0.0 (ABI version 105) - this is the exact
// version the nvoptix.dll bundled with the installed graphics driver ships (confirmed
// via its file version), which is NOT necessarily the same as the latest OptiX SDK you
// might have installed for headers (ABI is driver-runtime-bound, not SDK-download-
// bound - a newer SDK's ABI can be rejected with OPTIX_ERROR_UNSUPPORTED_ABI_VERSION
// even on a recent driver if the driver's bundled OptiX runtime hasn't caught up).
// Ground-truth headers live at
// "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" (optix_types.h for
// structs/enums, optix_function_table.h for the ABI version and OptixFunctionTable
// layout, internal/optix_device_impl.h for device-side intrinsic pseudo-call
// conventions used by OptixTrace.tt/OptixPayload.tt). When upgrading, first check
// nvoptix.dll's actual file version (Windows: right-click properties, or PowerShell
// (Get-Item <path>).VersionInfo) against the driver you're targeting - don't just grab
// the newest SDK - then diff every struct/enum/function-pointer in this project
// against the matching SDK's headers field-for-field - silent field/order drift here
// causes native heap corruption or cryptic driver-side compile errors, not C# compile
// errors. See docs/OPTIX_ROADMAP.md for the full audit checklist.

namespace ILGPU.OptiX
{
    partial class OptixAPI
    {
        #region Static

        [SuppressMessage("Naming", "CA1707:Identifiers should not contain underscores")]
        public const int OPTIX_ABI_VERSION = 105;

        private delegate OptixResult OptixQueryFunctionTable(
            int abiId,
            uint numOptions,
            IntPtr optionKeys,
            IntPtr optionValues,
            IntPtr functionTable,
            ulong sizeOfTable);

        #endregion

        #region Instance

        // OptixAPI.Current is a process-wide singleton, so Init()/Uninit() must be
        // ref-counted rather than unconditionally loading/freeing nvoptix.dll and
        // clearing functionTable/the delegate cache on every call - without this, a
        // second independent caller's Uninit() (e.g. a second OptixContext, or any
        // future multi-context/multi-window scenario) would free the module and zero
        // out functionTable out from under a still-live first caller's pipelines/
        // modules, which would then call through stale/zeroed function pointers.
        private readonly object initLock = new object();
        private int initRefCount;

        #endregion

        #region Methods

        /// <summary>
        /// Initializes the Optix API. Safe to call more than once (e.g. from multiple
        /// independent <see cref="OptixContext"/> users) - reference-counted against
        /// the matching number of <see cref="Uninit"/> calls.
        /// </summary>
        /// <returns>The OptiX result.</returns>
        public OptixResult Init()
        {
            lock (initLock)
            {
                if (initRefCount > 0)
                {
                    initRefCount++;
                    return OptixResult.OPTIX_SUCCESS;
                }

                hmodule = LoadOptixDLL();
                if (hmodule == IntPtr.Zero)
                    return OptixResult.OPTIX_ERROR_LIBRARY_NOT_FOUND;

                var proc = NativeMethods.GetProcAddressA(hmodule, "optixQueryFunctionTable");
                if (proc == IntPtr.Zero)
                {
                    NativeMethods.FreeLibrary(hmodule);
                    hmodule = IntPtr.Zero;
                    return OptixResult.OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
                }

                var functionTableSize = Marshal.SizeOf<OptixFunctionTable>();
                Debug.Assert(
                    functionTableSize == IntPtr.Size * 52,
                    "OptixFunctionTable field count no longer matches the native " +
                    "OptixFunctionTable (52 function pointers in OptiX SDK 9.0.0) - " +
                    "re-check optix_function_table.h. See docs/OPTIX_ROADMAP.md.");
                using var functionTablePtr = SafeHGlobal.Alloc(functionTableSize);

                var query =
                    Marshal.GetDelegateForFunctionPointer<OptixQueryFunctionTable>(proc);
                var result = query(
                    OPTIX_ABI_VERSION,
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero,
                    functionTablePtr,
                    (ulong)functionTableSize);

                if (result != OptixResult.OPTIX_SUCCESS)
                {
                    NativeMethods.FreeLibrary(hmodule);
                    hmodule = IntPtr.Zero;
                    return result;
                }

                functionTable = Marshal.PtrToStructure<OptixFunctionTable>(functionTablePtr);
                initRefCount = 1;
                return result;
            }
        }

        /// <summary>
        /// Uninitializes the OptiX API. Must be called once per successful
        /// <see cref="Init"/> call - the module is only actually unloaded once the
        /// matching number of <see cref="Uninit"/> calls has been made, so one
        /// caller's Uninit() cannot invalidate a still-live second caller's function
        /// pointers. A call with no matching outstanding Init() is a no-op.
        /// </summary>
        /// <returns>The OptiX result.</returns>
        public OptixResult Uninit()
        {
            lock (initLock)
            {
                if (initRefCount == 0)
                    return OptixResult.OPTIX_SUCCESS;

                if (--initRefCount > 0)
                    return OptixResult.OPTIX_SUCCESS;

                functionTable = default;
                // Cached delegates (see OptixAPI.cs's "Delegate cache" region) are only
                // valid for as long as the function pointers they wrap point into a
                // loaded nvoptix.dll - clear them here so a later Init() re-resolves
                // fresh ones against the (possibly different) reloaded module, rather
                // than calling into whatever the OS already unloaded/reused that
                // address for.
                ClearDelegateCache();

                if (hmodule != IntPtr.Zero)
                {
                    NativeMethods.FreeLibrary(hmodule);
                    hmodule = IntPtr.Zero;
                }

                return OptixResult.OPTIX_SUCCESS;
            }
        }

        /// <summary>
        /// Loads the OptiX DLL.
        /// </summary>
        /// <returns>A handle to the loaded DLL module.</returns>
        private static IntPtr LoadOptixDLL()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return LoadOptixWindowsDLL("nvoptix.dll");
            }
            else
            {
                return NativeMethods.LoadLibrary("libnvoptix.so.1");
            }
        }

        /// <summary>
        /// Loads the OptiX DLL on the Windows platform.
        /// </summary>
        /// <returns>A handle to the loaded DLL module.</returns>
        private static IntPtr LoadOptixWindowsDLL(string filename)
        {
            var systemPath = Path.Combine(Environment.SystemDirectory, filename);
            var handle = NativeMethods.LoadLibrary(systemPath);
            if (handle != IntPtr.Zero)
                return handle;

            return LoadOptixWindowsDLLFromConfigurationManager(filename);
        }


        /// <summary>
        /// Attempts to load the OptiX DLL from the Configuration Manager.
        /// </summary>
        /// <param name="filename">The name of the OptiX DLL to load.</param>
        /// <returns>A handle to the loaded DLL module.</returns>
        [SuppressMessage(
            "Design",
            "CA1031:Do not catch general exception types")]
        private static IntPtr LoadOptixWindowsDLLFromConfigurationManager(string filename)
        {
            // If we didn't find it, go looking in the register store.  Since nvoptix.dll
            // doesn't have its own registry entry, we are going to look for the opengl
            // driver which lives next to nvoptix.dll.  0 (null) will be returned if any
            // errors occured.
            foreach (var deviceName in GetDeviceNames())
            {
                try
                {
                    var driverPath = GetDeviceOpenGLDriverPath(deviceName);
                    var basePath = Path.GetDirectoryName(driverPath);
                    var dllPath = Path.Combine(basePath, filename);
                    var handle = NativeMethods.LoadLibrary(dllPath);
                    if (handle != IntPtr.Zero)
                        return handle;
                }
                catch (Exception)
                {
                    // Continue to the next device if errors are encountered.
                }
            }

            return IntPtr.Zero;
        }

        /// <summary>
        /// Enumerate the registry store for installed OpenGL devices.
        /// </summary>
        /// <returns>List of device names.</returns>
        private static IEnumerable<string> GetDeviceNames()
        {
            const string DeviceInstanceIdentifiersGUID =
                "{4d36e968-e325-11ce-bfc1-08002be10318}";
            const uint DeviceListFlags =
                NativeMethods.CM_GETIDLIST_FILTER_CLASS |
                NativeMethods.CM_GETIDLIST_FILTER_PRESENT;

            // Returns the required size to retrieve the device list.
            uint deviceListSize = 0;
            if (NativeMethods.CM_Get_Device_ID_List_Size(
                ref deviceListSize,
                DeviceInstanceIdentifiersGUID,
                DeviceListFlags)
                != NativeMethods.CR_SUCCESS)
            {
                return Array.Empty<string>();
            }

            // Returns a list of null-terminated strings.
            // The list itself is double null-terminated.
            var buffer = new char[deviceListSize];
            if (NativeMethods.CM_Get_Device_ID_List(
                DeviceInstanceIdentifiersGUID,
                buffer,
                deviceListSize,
                DeviceListFlags)
                != NativeMethods.CR_SUCCESS)
            {
                return Array.Empty<string>();
            }

            var bufferStr = new string(buffer);
            return bufferStr.Split(new[] { '\0' }, StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>
        /// Retrieves the path to the OpenGL driver DLL for the specified device.
        /// </summary>
        /// <param name="deviceName">The device name.</param>
        /// <returns>The path to the driver DLL.</returns>
        private static string GetDeviceOpenGLDriverPath(string deviceName)
        {
            const string ValueName = "OpenGLDriverName";

            var devNode = IntPtr.Zero;
            if (NativeMethods.CM_Locate_DevNode(
                ref devNode,
                deviceName,
                NativeMethods.CM_LOCATE_DEVNODE_NORMAL)
                != NativeMethods.CR_SUCCESS)
            {
                return string.Empty;
            }

            if (NativeMethods.CM_Open_DevNode_Key(
                devNode,
                NativeMethods.KEY_QUERY_VALUE,
                0,
                NativeMethods.RegDisposition_OpenExisting,
                out var regKeyPtr,
                NativeMethods.CM_REGISTRY_SOFTWARE)
                != NativeMethods.CR_SUCCESS)
            {
                return string.Empty;
            }

            using var regKeyHandle = new SafeRegistryHandle(regKeyPtr, ownsHandle: true);
            var regKey = RegistryKey.FromHandle(regKeyHandle);
            var valueKind = regKey.GetValueKind(ValueName);
            if (valueKind == RegistryValueKind.MultiString)
            {
                string[]? paths = ((string[]?)regKey.GetValue(ValueName));
                if (paths == null)
                    return string.Empty;
                return paths[0];
            }
            else if (valueKind == RegistryValueKind.String)
            {
                string? path = (string?)regKey.GetValue(ValueName);
                return path ?? string.Empty;
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        #endregion

        #region Native Methods

        private static partial class NativeMethods
        {
            #region Library Loader

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern IntPtr LoadLibrary(string lpFileName);

            [DllImport(
                "kernel32.dll",
                EntryPoint = "GetProcAddress",
                CharSet = CharSet.Ansi)]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            [SuppressMessage(
                "Globalization",
                "CA2101:Specify marshaling for P/Invoke string arguments",
                Justification = "OptiX does not work with GetProcAddressW")]
            public static extern IntPtr GetProcAddressA(
                IntPtr hModule,
                string lpProcName);

            [DllImport("kernel32.dll")]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern bool FreeLibrary(IntPtr hModule);

            #endregion

            #region Configuration Manager

            /// <summary>
            /// Configuration Manager CONFIGRET return status codes.
            /// </summary>
            public const uint CR_SUCCESS = 0x00000000;

            /// <summary>
            /// Flags for CM_Get_Device_ID_List, CM_Get_Device_ID_List_Size.
            /// </summary>
            public const uint CM_GETIDLIST_FILTER_PRESENT = 0x00000100;
            public const uint CM_GETIDLIST_FILTER_CLASS = 0x00000200;

            /// <summary>
            /// Flags for CM_Locate_DevNode.
            /// </summary>
            public const uint CM_LOCATE_DEVNODE_NORMAL = 0x00000000;

            /// <summary>
            /// Registry disposition values.
            /// (specified in call to CM_Open_DevNode_Key and CM_Open_Class_Key).
            /// </summary>
            public const uint RegDisposition_OpenExisting = 0x00000001;

            /// <summary>
            /// Registry Branch Locations (for CM_Open_DevNode_Key).
            /// </summary>
            public const uint CM_REGISTRY_SOFTWARE = 0x00000001;

            /// <summary>
            /// Registry Specific Access Rights.
            /// </summary>
            public const uint KEY_QUERY_VALUE = 0x0001;

            [DllImport("setupapi.dll", CharSet = CharSet.Unicode)]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern int CM_Get_Device_ID_List_Size(
                ref uint idListlen,
                string lpFilter,
                uint ulFlags);

            [DllImport("setupapi.dll", CharSet = CharSet.Unicode)]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern int CM_Get_Device_ID_List(
                string lpFilter,
                char[] buffer,
                uint bufferLen,
                uint ulFlags);

            [DllImport("setupapi.dll", CharSet = CharSet.Unicode)]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern int CM_Locate_DevNode(
                ref IntPtr devNode,
                string deviceName,
                uint ulFlags);

            [DllImport("setupapi.dll")]
            [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
            public static extern int CM_Open_DevNode_Key(
                IntPtr devNode,
                uint samDesired,
                ulong ulHardwareProfile,
                ulong Disposition,
                out IntPtr phkDevice,
                uint ulFlags);

            #endregion
        }

        #endregion
    }
}
