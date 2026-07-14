# ILGPU.OptiX

.NET bindings for [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) built on top of [ILGPU](https://github.com/m4rs-mt/ILGPU), so ray-tracing raygen/miss/hit programs are written and JIT-compiled as ordinary C# - no separate CUDA/OptiX C++ toolchain required.

Copyright (c) 2020-2022 ILGPU Project. All rights reserved.

## Requirements

- An NVIDIA RTX-capable GPU and driver with OptiX support installed.
- The [NVIDIA OptiX SDK](https://developer.nvidia.com/designworks/optix/downloads/legacy) installed locally (the library resolves `nvoptix.dll` from the driver at runtime; the SDK is only needed for header/ABI reference during development).
- .NET (see `Src/ILGPU.OptiX/ILGPU.OptiX.csproj` for the exact target frameworks).

## Quick start

```csharp
using var context = Context.Create(b => b.Cuda());
using var accelerator = context.CreateCudaAccelerator(0);
using var rt = OptixRayTracer.Create(accelerator);

using var pipeline = rt.CreatePipeline<LaunchParams>(b => b
    .Raygen(RenderFrame)
    .RayType("radiance", r => r
        .Payload<RadiancePayload>()
        .Miss(MissRadiance)
        .HitGroup<MaterialData>(closestHit: ClosestHitRadiance))
    .MaxTraceDepth(2));

pipeline.SetHitRecords<MaterialData>(materials);
pipeline.Launch(launchParams, width, height);
```

`OptixRayTracer` and `RayTracingPipeline<T>` (in `ILGPU.OptiX.Pipeline`) own the module/pipeline compile options, SBT record packing, stack size computation, and a persistent launch-params buffer, so a working pipeline no longer requires hand-picked compile-option structs, hand-measured SBT record sizes, or magic stack-size numbers. The lower-level APIs these are built on (`CreateModule`, `CreateProgramGroup`, raw `OptixShaderBindingTable`, etc.) remain public for cases the facade doesn't cover yet.

## Samples

`Samples/Sample01` through `Samples/Sample15` are a tutorial progression from "initialize the library" through a full interactive PBR path tracer, each adding one concept (acceleration structures, textures, multiple ray types, denoising, instancing, curves). See [`tutorials/readme.md`](tutorials/readme.md) for the walkthroughs written so far.
