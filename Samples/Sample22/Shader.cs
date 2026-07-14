// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Shader.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using OpenTK.Graphics.OpenGL4;
using System;

namespace Sample22
{
    // Minimal GLSL program wrapper - same compile/link pattern as Sample14's own
    // Rendering/Shader.cs, trimmed to just what the fullscreen display quad needs.
    public sealed class Shader : IDisposable
    {
        public int Handle { get; }

        public Shader(string vertexSource, string fragmentSource)
        {
            int vertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(vertexShader, vertexSource);
            CompileShader(vertexShader);

            int fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, fragmentSource);
            CompileShader(fragmentShader);

            Handle = GL.CreateProgram();
            GL.AttachShader(Handle, vertexShader);
            GL.AttachShader(Handle, fragmentShader);
            LinkProgram(Handle);

            GL.DetachShader(Handle, vertexShader);
            GL.DetachShader(Handle, fragmentShader);
            GL.DeleteShader(vertexShader);
            GL.DeleteShader(fragmentShader);
        }

        static void CompileShader(int shader)
        {
            GL.CompileShader(shader);
            GL.GetShader(shader, ShaderParameter.CompileStatus, out var status);
            if (status != (int)All.True)
                throw new InvalidOperationException($"Shader compile error: {GL.GetShaderInfoLog(shader)}");
        }

        static void LinkProgram(int program)
        {
            GL.LinkProgram(program);
            GL.GetProgram(program, GetProgramParameterName.LinkStatus, out var status);
            if (status != (int)All.True)
                throw new InvalidOperationException($"Shader link error: {GL.GetProgramInfoLog(program)}");
        }

        public void Use() => GL.UseProgram(Handle);

        public void SetInt(string name, int value) => GL.Uniform1(GL.GetUniformLocation(Handle, name), value);

        public void Dispose() => GL.DeleteProgram(Handle);
    }
}
