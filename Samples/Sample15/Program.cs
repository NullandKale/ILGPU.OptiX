using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

namespace Sample15
{
    public static class Program
    {
        public static void Main()
        {
            var gameWindowSettings = GameWindowSettings.Default;
            var nativeWindowSettings = new NativeWindowSettings
            {
                ClientSize = new Vector2i(1200, 800),
                Title = "Sample15 - PBR Path Tracer",
                Flags = ContextFlags.ForwardCompatible,
                Profile = ContextProfile.Core,
                APIVersion = new System.Version(3, 3),
            };

            using var window = new RenderWindow(gameWindowSettings, nativeWindowSettings);
            window.Run();
        }
    }
}
