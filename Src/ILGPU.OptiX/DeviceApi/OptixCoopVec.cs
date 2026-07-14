// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCoopVec.tt/OptixCoopVec.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------








using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the device-side cooperative-vector intrinsic surface: matrix-vector multiply, the elementwise ops,
    /// and the two training primitives (reduce-sum-accumulate, outer-product-accumulate)
    /// - the building blocks of an MLP evaluated inline in a hit/raygen program (e.g. a
    /// neural material), forward and backward pass alike.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The OptiX SDK's own C++ header (internal/optix_device_impl_coop_vec.h) exposes
    /// two calling conventions per op: a register-packed form (cooperative vectors held
    /// in up to 64 PTX registers) and a pointer-passing fallback ("_ptr" suffixed
    /// pseudo-calls, e.g. "_optix_matvecmul_ptr") where every vector/matrix/bias is a
    /// device address + element type + count. This binding wraps the "_ptr" family - the
    /// register-packed form needs 48-69 EmitRef arguments per call even at its smallest
    /// (16-element) tier, which exceeds ILGPU's own 44-argument <c>CudaAsm.EmitRef</c>
    /// ceiling (confirmed via reflection against the actual referenced ILGPU 1.5.3
    /// package) for every
    /// op with more than one vector operand (MatVecMul, the 2-input and 3-input
    /// elementwise ops) - not implementable, not a convenience choice.
    /// </para>
    /// <para>
    /// <b>Every non-pointer, non-byte-offset argument must be a genuine PTX compile-time
    /// constant</b> - discovered via a real GPU compile failure ("vector op is not
    /// constant" / "Output vector element type is not constant"), not assumed. This
    /// mirrors the real C++ API exactly (every one of these is a template parameter in
    /// <c>optix_device_impl_coop_vec.h</c>, e.g. <c>optixCoopVecMatMul&lt;VecTOut, VecTIn,
    /// inputInterpretation, matrixLayout, transpose, N, K, matrixElementType,
    /// biasElementType&gt;</c> - only the device pointers and byte offsets are ordinary
    /// runtime function parameters there too). A plain <c>CudaAsm.EmitRef Input&lt;T&gt;</c>
    /// round-trips its value through a local memory store+load (see
    /// <see cref="OptixTrace"/>'s own class doc comment for the same finding on
    /// optixTrace's payload-type/count arguments) rather than a direct "mov", so it does
    /// not satisfy this. The fix is the same one <see cref="OptixTrace"/> already
    /// established: every constant-typed operand gets an <c>Output&lt;uint&gt;</c> slot
    /// (a fresh register, not memory-backed) whose value is assigned via a literal
    /// "mov.u32 %N, IMMEDIATE;" instruction textually prefixed onto the call, where the
    /// immediate is a genuine C# string literal baked in at this template's generation
    /// time - which is why this file is T4-generated (one overload per required constant
    /// combination) rather than hand-written with runtime parameters, unlike
    /// <see cref="OptixReorder"/>/<see cref="OptixHitObjectQueries"/>'s single-combination
    /// calls.
    /// </para>
    /// <para>
    /// Only <see cref="OptixCoopVecElemType.Float16"/> element type is generated (every
    /// other type would multiply the already-large generated overload count further, and
    /// nothing in this codebase needs them yet); <c>MatVecMul</c> is generated for
    /// row/column counts 1-8, InferencingOptimal layout, no transpose (the
    /// standard inference case); the elementwise ops for vector sizes 1-8.
    /// Bump <c>MaxSize</c> in the .tt template (and regenerate via <c>dotnet-t4</c>) if a
    /// future consumer needs larger layers or other element types.
    /// </para>
    /// <para>
    /// A curated set of additional overloads is generated on top of the 1-8
    /// loops above (additive - the loops' own output is unchanged) for Sample25's
    /// 64-wide NRC MLP layers, since uniformly bumping <c>MaxSize</c> to 64 would explode
    /// <c>MatVecMul</c>/<c>OuterProductAccumulate</c> to 64x64 = 4096 overloads each
    /// (mostly unused): <c>MatVecMul_N64_K64</c>/<c>_N3_K64</c> (InferencingOptimal, no
    /// transpose - inference against the EMA weight snapshot), <c>..._Training</c>
    /// (TrainingOptimal, no transpose - forward pass against the live training weights),
    /// and <c>..._TrainingTranspose</c> (TrainingOptimal, transpose = true - propagates a
    /// gradient backward through the same layer's stored matrix without a separate
    /// transposed copy); <c>OuterProductAccumulate_SA64_SB64</c>/<c>_SA3_SB64</c>; and
    /// size-64 elementwise/<c>Load</c>/<c>ReduceSumAccumulate</c> overloads. See the
    /// curated-dims variables declared alongside <c>MaxSize</c> above.
    /// </para>
    /// </remarks>
    [CLSCompliant(false)]
    public static class OptixCoopVec
    {

        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 1x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 1-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 1-element result vector to.</param>
        public static void MatVecMul_N1_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 1; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 2x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 2-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 2-element result vector to.</param>
        public static void MatVecMul_N2_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 2; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 3x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 4x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 4-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 4-element result vector to.</param>
        public static void MatVecMul_N4_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 4; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 5x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 5-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 5-element result vector to.</param>
        public static void MatVecMul_N5_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 5; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 6x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 6-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 6-element result vector to.</param>
        public static void MatVecMul_N6_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 6; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 7x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 7-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 7-element result vector to.</param>
        public static void MatVecMul_N7_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 7; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x1 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 1-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K1(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 1; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x2 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 2-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K2(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 2; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x3 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K3(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 3; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x4 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 4-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K4(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 4; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x5 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 5-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K5(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 5; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x6 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 6-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K6(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 6; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x7 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 7-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K7(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 7; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a
        /// 8x8 Float16 weight matrix (InferencingOptimal layout, no
        /// transpose) - a single fully connected layer. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 8-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 8-element bias vector (16-byte aligned).</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 8-element result vector to.</param>
        public static void MatVecMul_N8_K8(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %4, 10753; mov.u32 %5, 8; " +
                "mov.u32 %6, 8; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 64x64
        /// Float16 weight matrix (InferencingOptimal layout).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 64-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 64-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 64-element result vector to.</param>
        public static void MatVecMul_N64_K64(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %4, 10753; mov.u32 %5, 64; " +
                "mov.u32 %6, 64; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 64x64
        /// Float16 weight matrix (TrainingOptimal layout).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 64-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 64-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 64-element result vector to.</param>
        public static void MatVecMul_N64_K64_Training(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %4, 10753; mov.u32 %5, 64; " +
                "mov.u32 %6, 64; mov.u32 %10, 10819; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 64x64
        /// Float16 weight matrix (TrainingOptimal layout, transposed - propagates a gradient through a fully connected layer during backprop, reusing the same untransposed stored matrix).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 64-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 64-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 64-element result vector to.</param>
        public static void MatVecMul_N64_K64_TrainingTranspose(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %4, 10753; mov.u32 %5, 64; " +
                "mov.u32 %6, 64; mov.u32 %10, 10819; mov.u32 %11, 1; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 3x64
        /// Float16 weight matrix (InferencingOptimal layout).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 64-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K64(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 64; mov.u32 %10, 10818; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 3x64
        /// Float16 weight matrix (TrainingOptimal layout).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 64-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 3-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 3-element result vector to.</param>
        public static void MatVecMul_N3_K64_Training(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 64; mov.u32 %10, 10819; mov.u32 %11, 0; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>
        /// Computes <c>outputVector = matrix * inputVector + bias</c> for a 3x64
        /// Float16 weight matrix (TrainingOptimal layout, transposed - propagates a gradient through a fully connected layer during backprop, reusing the same untransposed stored matrix).
        /// Curated overload for Sample25 (NRC's 64-wide layers) - see the curated-dims
        /// comment above <c>MaxSize</c>. Wraps <c>_optix_matvecmul_ptr</c>.
        /// </summary>
        /// <param name="inputVector">Device address of the 3-element input vector.</param>
        /// <param name="matrix">Device address of the weight matrix (64-byte aligned).</param>
        /// <param name="matrixOffsetInBytes">Byte offset into <paramref name="matrix"/> (64-byte aligned).</param>
        /// <param name="bias">Device address of the 64-element bias vector (16-byte aligned) - point at a zero-filled buffer for a no-bias layer.</param>
        /// <param name="biasOffsetInBytes">Byte offset into <paramref name="bias"/> (16-byte aligned).</param>
        /// <param name="outputVector">Device address to write the 64-element result vector to.</param>
        public static void MatVecMul_N3_K64_TrainingTranspose(
            ulong inputVector,
            ulong matrix,
            uint matrixOffsetInBytes,
            ulong bias,
            uint biasOffsetInBytes,
            ulong outputVector)
        {
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Output<uint> _inputInterpretation = default;
            Output<uint> _n = default;
            Output<uint> _k = default;
            Input<ulong> _matrix = matrix;
            Input<uint> _matrixOffsetInBytes = matrixOffsetInBytes;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Output<uint> _matrixLayout = default;
            Output<uint> _transpose = default;
            Output<uint> _matrixElementType = default;
            Input<ulong> _bias = bias;
            Input<uint> _biasOffsetInBytes = biasOffsetInBytes;
            Output<uint> _biasElementType = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %4, 10753; mov.u32 %5, 3; " +
                "mov.u32 %6, 64; mov.u32 %10, 10819; mov.u32 %11, 1; " +
                "mov.u32 %12, 10753; mov.u32 %15, 10753; " +
                "call (), _optix_matvecmul_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);",
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputInterpretation,
                ref _n,
                ref _k,
                ref _matrix,
                ref _matrixOffsetInBytes,
                ref _rowColumnStrideInBytes,
                ref _matrixLayout,
                ref _transpose,
                ref _matrixElementType,
                ref _bias,
                ref _biasOffsetInBytes,
                ref _biasElementType,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (1-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S1(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (2-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S2(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (3-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S3(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (4-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S4(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (5-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S5(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (6-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S6(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (7-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S7(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (8-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S8(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise 2^x. (64-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Exp2_S64(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10785; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (1-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S1(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (2-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S2(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (3-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S3(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (4-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S4(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (5-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S5(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (6-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S6(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (7-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S7(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (8-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S8(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise log2(x). (64-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Log2_S64(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10786; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (1-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S1(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (2-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S2(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (3-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S3(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (4-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S4(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (5-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S5(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (6-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S6(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (7-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S7(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (8-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S8(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise hyperbolic tangent - the activation used between MLP layers. (64-element Float16 vectors.) Wraps <c>_optix_vector_op1_ptr</c>.</summary>
        public static void Tanh_S64(ulong inputVector, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVector = inputVector;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10787; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op1_ptr, (%0,%1,%2,%3,%4,%5,%6);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVector,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise max(a, b) - e.g. ReLU via a zero-filled second vector. (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Max_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10788; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise min(a, b). (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Min_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10789; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a * b. (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Mul_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10791; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a + b. (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Add_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10792; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise a - b. (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Sub_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10793; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (1-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S1(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (2-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S2(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (3-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S3(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (4-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S4(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (5-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S5(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (6-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S6(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (7-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S7(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (8-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S8(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise step(a, b) - 1 where b &gt;= a, 0 otherwise. (64-element Float16 vectors.) Wraps <c>_optix_vector_op2_ptr</c>.</summary>
        public static void Step_S64(ulong inputVectorA, ulong inputVectorB, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10795; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op2_ptr, (%0,%1,%2,%3,%4,%5,%6,%7);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (1-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S1(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 1; " +
                "mov.u32 %3, 10753; mov.u32 %4, 1; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (2-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S2(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 2; " +
                "mov.u32 %3, 10753; mov.u32 %4, 2; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (3-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S3(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 3; " +
                "mov.u32 %3, 10753; mov.u32 %4, 3; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (4-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S4(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 4; " +
                "mov.u32 %3, 10753; mov.u32 %4, 4; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (5-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S5(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 5; " +
                "mov.u32 %3, 10753; mov.u32 %4, 5; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (6-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S6(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 6; " +
                "mov.u32 %3, 10753; mov.u32 %4, 6; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (7-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S7(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 7; " +
                "mov.u32 %3, 10753; mov.u32 %4, 7; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (8-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S8(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 8; " +
                "mov.u32 %3, 10753; mov.u32 %4, 8; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>Elementwise fused multiply-add: a * b + c (64-element Float16 vectors). Wraps <c>_optix_vector_op3_ptr</c>.</summary>
        public static void FFma_S64(ulong inputVectorA, ulong inputVectorB, ulong inputVectorC, ulong outputVector)
        {
            Output<uint> _op = default;
            Output<uint> _outputElementType = default;
            Output<uint> _outputSize = default;
            Output<uint> _inputElementType = default;
            Output<uint> _inputSize = default;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;
            Input<ulong> _inputVectorC = inputVectorC;
            Input<ulong> _outputVector = outputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10790; mov.u32 %1, 10753; mov.u32 %2, 64; " +
                "mov.u32 %3, 10753; mov.u32 %4, 64; " +
                "call (), _optix_vector_op3_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8);",
                ref _op,
                ref _outputElementType,
                ref _outputSize,
                ref _inputElementType,
                ref _inputSize,
                ref _inputVectorA,
                ref _inputVectorB,
                ref _inputVectorC,
                ref _outputVector);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (1-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S1(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (2-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S2(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (3-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S3(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (4-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S4(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (5-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S5(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (6-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S6(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (7-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S7(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (8-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S8(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Loads/copies raw device data at <paramref name="sourceAddress"/> into
        /// <paramref name="destinationVector"/> (64-element Float16). Wraps
        /// <c>_optix_vector_load_ptr</c>.
        /// </summary>
        public static void Load_S64(ulong sourceAddress, ulong destinationVector)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _source = sourceAddress;
            Input<ulong> _destination = destinationVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; " +
                "call (), _optix_vector_load_ptr, (%0,%1,%2,%3);",
                ref _elementType,
                ref _size,
                ref _source,
                ref _destination);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (1-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S1(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (2-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S2(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (3-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S3(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (4-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S4(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (5-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S5(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (6-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S6(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (7-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S7(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (8-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S8(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the elementwise sum-reduction of <paramref name="inputVector"/>
        /// (64-element Float16) into <paramref name="outputVector"/> at
        /// <paramref name="offsetInBytes"/> - the bias-gradient primitive of
        /// backpropagation. Wraps <c>_optix_reduce_sum_accumulate_ptr</c>.
        /// </summary>
        public static void ReduceSumAccumulate_S64(ulong inputVector, ulong outputVector, uint offsetInBytes)
        {
            Output<uint> _elementType = default;
            Output<uint> _size = default;
            Input<ulong> _outputVector = outputVector;
            Input<uint> _offsetInBytes = offsetInBytes;
            Input<ulong> _inputVector = inputVector;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; " +
                "call (), _optix_reduce_sum_accumulate_ptr, (%0,%1,%2,%3,%4);",
                ref _elementType,
                ref _size,
                ref _outputVector,
                ref _offsetInBytes,
                ref _inputVector);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (1x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA1_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 1; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (2x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA2_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 2; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (4x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA4_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 4; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (5x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA5_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 5; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (6x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA6_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 6; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (7x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA7_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 7; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x1, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB1(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 1; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x2, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB2(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 2; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x3, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB3(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 3; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x4, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB4(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 4; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x5, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB5(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 5; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x6, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB6(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 6; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x7, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB7(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 7; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (8x8, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Wraps <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA8_SB8(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 8; mov.u32 %2, 10753; " +
                "mov.u32 %3, 8; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (64x64, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Curated overload for Sample25 (NRC's 64-wide layers) - see the
        /// curated-dims comment above <c>MaxSize</c>. Wraps
        /// <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA64_SB64(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 64; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


        /// <summary>
        /// Accumulates (adds) the outer product <c>vecA * vecB^T</c>
        /// (3x64, Float16, TrainingOptimal layout) into the matrix
        /// at <paramref name="outputMatrix"/> - the weight-gradient primitive of
        /// backpropagation. Curated overload for Sample25 (NRC's 64-wide layers) - see the
        /// curated-dims comment above <c>MaxSize</c>. Wraps
        /// <c>_optix_outer_product_accumulate_ptr</c>.
        /// </summary>
        public static void OuterProductAccumulate_SA3_SB64(
            ulong inputVectorA, ulong inputVectorB, ulong outputMatrix, uint offsetInBytes)
        {
            Output<uint> _elementTypeA = default;
            Output<uint> _sizeA = default;
            Output<uint> _elementTypeB = default;
            Output<uint> _sizeB = default;
            Input<ulong> _outputMatrix = outputMatrix;
            Input<uint> _offsetInBytes = offsetInBytes;
            Output<uint> _matrixLayout = default;
            Input<uint> _rowColumnStrideInBytes = 0u;
            Input<ulong> _inputVectorA = inputVectorA;
            Input<ulong> _inputVectorB = inputVectorB;

            CudaAsm.EmitRef(
                "mov.u32 %0, 10753; mov.u32 %1, 3; mov.u32 %2, 10753; " +
                "mov.u32 %3, 64; mov.u32 %6, 10819; " +
                "call (), _optix_outer_product_accumulate_ptr, (%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);",
                ref _elementTypeA,
                ref _sizeA,
                ref _elementTypeB,
                ref _sizeB,
                ref _outputMatrix,
                ref _offsetInBytes,
                ref _matrixLayout,
                ref _rowColumnStrideInBytes,
                ref _inputVectorA,
                ref _inputVectorB);
        }


    }
}
