namespace Sample05
{
    public unsafe struct LaunchParams
    {
        public int FrameID;
        public uint* ColorBuffer;
        public Camera camera;
        public ulong traversable;
        public Vec3* vertices;
        public Vec3i* indices;
        public Vec3* primitiveColors;
    }
}
