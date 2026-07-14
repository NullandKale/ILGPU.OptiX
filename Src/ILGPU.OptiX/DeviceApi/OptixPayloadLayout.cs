// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPayloadLayout.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Host-side-only validation for payload structs used with
    /// <see cref="OptixTrace.Trace{T}"/> / <see cref="OptixPayload.Read{T}"/> /
    /// <see cref="OptixPayload.Write{T}"/>. Never call this from device code - it uses
    /// reflection, which ILGPU cannot compile.
    /// </summary>
    public static class OptixPayloadLayout
    {
        /// <summary>
        /// The number of 32-bit payload registers <typeparamref name="T"/> occupies -
        /// the value <c>NumPayloadValues</c> in the pipeline compile options must be
        /// set to for every ray type using this payload struct. Throws if
        /// <typeparamref name="T"/> is not exactly reinterpretable as that many
        /// consecutive float/int/uint registers (i.e. if it contains a field of any
        /// other type, or if compiler-inserted padding would make a bit-for-bit
        /// register mapping unsound).
        /// </summary>
        public static int CountOf<T>() where T : unmanaged
        {
            var type = typeof(T);
            int leafFieldCount = CountLeafFields(type);
            int expectedSize = leafFieldCount * sizeof(uint);
            int actualSize = Unsafe.SizeOf<T>();

            if (actualSize != expectedSize)
                throw new ArgumentException(
                    $"Payload struct {type.Name} is {actualSize} bytes but its " +
                    $"float/int/uint fields only account for {expectedSize} bytes - " +
                    "the difference is compiler-inserted padding, which would silently " +
                    "misalign payload registers between the raygen Trace<T>() call and " +
                    "hit/miss program Read<T>()/Write<T>() calls. Reorder fields (larger " +
                    "types first) or split into two payload structs.");

            if (leafFieldCount > OptixPayload.MaxCount)
                throw new ArgumentException(
                    $"Payload struct {type.Name} needs {leafFieldCount} registers, " +
                    $"exceeding the {OptixPayload.MaxCount}-register ceiling (see " +
                    $"{nameof(OptixPayload)}.{nameof(OptixPayload.MaxCount)}).");

            return leafFieldCount;
        }

        private static int CountLeafFields(Type type)
        {
            int count = 0;
            foreach (var field in type.GetFields(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            {
                var fieldType = field.FieldType;
                if (fieldType == typeof(float) || fieldType == typeof(int) || fieldType == typeof(uint))
                {
                    count++;
                }
                else if (fieldType.IsValueType && !fieldType.IsPrimitive && !fieldType.IsEnum)
                {
                    count += CountLeafFields(fieldType);
                }
                else
                {
                    throw new ArgumentException(
                        $"{type.FullName} contains field '{field.Name}' of type " +
                        $"{fieldType.Name}; payload structs may only contain float/int/" +
                        "uint fields, or structs composed of them (e.g. " +
                        "System.Numerics.Vector3).");
                }
            }
            return count;
        }
    }
}
