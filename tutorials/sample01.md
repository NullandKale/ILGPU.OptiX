# ILGPU.Optix Sample 01

Sample 01 is a very basic program consisting of only a few lines of code. The goal of this program is to ensure your build environment is setup correctly, and your GPU supports OptiX

```csharp
using ILGPU;
using ILGPU.OptiX;
using ILGPU.Runtime.Cuda;
using System;

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
```

I want to call attention to two key lines which are sandwiched between the Console.Writelines:

```csharp
using var context = Context.Create(b => b.Cuda());
```

The first line uses the ILGPU context builder pattern to build an ILGPU context object. This is used to enable features like optimization or fast math or the algorithms library. In this case to enable the OptiX extensions, and force the accelerators to only be CUDA accelerators.

```csharp
using var accelerator = context.CreateCudaAccelerator(0);
```

This line initializes a cuda context using the first gpu in your system.

# The Context and Accelerators

The ILGPU Context object is the root ILGPU object, its main purpose is to list what accelerators are available and allow you to create them at runtime.

The Accelerator object is a far more important. This object is used extensively for kernel compilation, memory allocation, stream management, ect. This is the main way you interact with the GPU.

Both of these objects are expensive to create, and ideally should only be created once when you start up your program. While it is possible and in some cases useful to destroy these objects and recreate them, doing so invalidates all kernels and memory buffers that were allocated using the accelerator / context. Recreating all of that stuff is slow and something we want to avoid.

Note that both lines use the using syntax! Both the ILGPU context object and the accelerator object are IDisposeable. It is a memory and resource leak to leave these objects undisposed. 

# Setup

If you want to replicate this sample from scratch follow these steps:

```
# 1. Create a new C# project
dotnet new console -n MyOptiXRayTracer -f net8.0

# 2. Install NuGet packages for ILGPU and ILGPU.Algorithms
dotnet add MyOptiXRayTracer/MyOptiXRayTracer.csproj package ILGPU
dotnet add MyOptiXRayTracer/MyOptiXRayTracer.csproj package ILGPU.Algorithms

# 3. Clone the external repository
git clone https://github.com/NullandKale/ILGPU.OptiX.git

# 4. Link the cloned project as a dependency
dotnet add MyOptiXRayTracer/MyOptiXRayTracer.csproj reference ILGPU.OptiX/Src/ILGPU.OptiX/ILGPU.OptiX.csproj

# 5. Optionally create a .sln for Visual Studio
dotnet new sln -n MyOptiXRayTracer
dotnet sln MyOptiXRayTracer.sln add MyOptiXRayTracer/MyOptiXRayTracer.csproj
dotnet sln MyOptiXRayTracer.sln add ILGPU.OptiX/Src/ILGPU.OptiX/ILGPU.OptiX.csproj
```

Eventually ILGPU.Optix will be a nuget package and you will not have to clone the entire repo. For now this will work, and it at least gives you the OptiX code for reference which is nice.

