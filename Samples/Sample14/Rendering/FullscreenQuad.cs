using OpenTK.Graphics.OpenGL4;
using System;

namespace Sample14
{
    /// <summary>
    /// Draws a display texture as a fullscreen quad - trivial passthrough shader since
    /// tonemap/gamma already happened on the GPU (TonemapKernel.tonemapAndFlip writes
    /// final display-ready BGRA bytes; the test pattern kernel does the same for M1).
    /// No depth test/blend/culling needed - one opaque fullscreen draw per frame.
    /// </summary>
    public sealed class FullscreenQuad : IDisposable
    {
        const string VertexSource = @"
#version 330 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main()
{
    vUv = aUv;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}";

        const string FragmentSource = @"
#version 330 core
in vec2 vUv;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main()
{
    FragColor = texture(screenTexture, vUv);
}";

        // Position (x,y) + UV (u,v) per vertex, quad covering NDC [-1,1] with V=0 at
        // the top row to match TonemapKernel.tonemapAndFlip's own row flip (it writes
        // dest already flipped so row 0 is the top of the image).
        static readonly float[] Vertices =
        {
            -1f,  1f, 0f, 0f,
             1f,  1f, 1f, 0f,
             1f, -1f, 1f, 1f,
            -1f, -1f, 0f, 1f,
        };

        static readonly uint[] Indices = { 0, 1, 2, 0, 2, 3 };

        readonly Shader shader;
        readonly int vao;
        readonly int vbo;
        readonly int ebo;

        public FullscreenQuad()
        {
            shader = new Shader(VertexSource, FragmentSource);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);

            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, Vertices.Length * sizeof(float), Vertices, BufferUsageHint.StaticDraw);

            ebo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
            GL.BufferData(BufferTarget.ElementArrayBuffer, Indices.Length * sizeof(uint), Indices, BufferUsageHint.StaticDraw);

            const int stride = 4 * sizeof(float);
            GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, stride, 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, stride, 2 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            GL.BindVertexArray(0);

            shader.Use();
            shader.SetInt("screenTexture", 0);
        }

        public void Draw(int glTextureHandle)
        {
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, glTextureHandle);

            shader.Use();
            GL.BindVertexArray(vao);
            GL.DrawElements(PrimitiveType.Triangles, Indices.Length, DrawElementsType.UnsignedInt, 0);
            GL.BindVertexArray(0);
        }

        public void Dispose()
        {
            GL.DeleteBuffer(vbo);
            GL.DeleteBuffer(ebo);
            GL.DeleteVertexArray(vao);
            shader.Dispose();
        }
    }
}
