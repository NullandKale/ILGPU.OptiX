using ILGPU.OptiX.Device;

namespace Sample05
{
    public struct LaunchParams
    {
        public int FrameID;
        public OptixDeviceView<uint> ColorBuffer;
        public Camera camera;
        public ulong traversable;
        public OptixDeviceView<Vec3> vertices;
        public OptixDeviceView<Vec3i> indices;
        public OptixDeviceView<Vec3> primitiveColors;
    }
}
