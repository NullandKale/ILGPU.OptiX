// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixLogCallback.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Native
{
    /// <summary>
    /// Matches OptiX's OptixLogCallback native function pointer type
    /// (void(*)(unsigned int level, const char* tag, const char* message, void* cbdata)).
    /// Assign OptixDeviceContextOptions.LogCallbackFunction via
    /// Marshal.GetFunctionPointerForDelegate on an instance of this delegate, and keep
    /// that delegate instance alive (e.g. as a field) for the device context's whole
    /// lifetime - the GC has no visibility into the native function pointer reference,
    /// so a delegate that's only a local variable can be collected while OptiX still
    /// holds the pointer.
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    [CLSCompliant(false)]
    public delegate void OptixLogCallback(uint level, IntPtr tag, IntPtr message, IntPtr callbackData);
}
