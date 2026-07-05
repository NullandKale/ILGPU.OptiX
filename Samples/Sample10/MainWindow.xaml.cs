using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Sample10
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public int width = 1200;
        public int height = 800;
        public WriteableBitmap wBitmap;
        public Int32Rect rect;

        public SampleRenderer sampleRenderer;
        public Thread renderThread;

        public bool run = true;

        // Mouse-drag camera controls: left = orbit, right = dolly, middle = pan.
        private Point? dragLastPos;
        private MouseButton? dragButton;

        public MainWindow()
        {
            InitializeComponent();

            Console.WriteLine("Controls: Left-drag orbit, Right-drag dolly (zoom), Middle-drag pan.");
            Console.WriteLine("Arrow keys move the point light in the XZ plane, PageUp/PageDown move it vertically");
            Console.WriteLine("(the window title updates with the light's new position).");

            wBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            rect = new Int32Rect(0, 0, width, height);
            Frame.Source = wBitmap;

            sampleRenderer = new SampleRenderer(width, height, this);
            Closing += MainWindow_Closing;

            Frame.MouseDown += Frame_MouseDown;
            Frame.MouseMove += Frame_MouseMove;
            Frame.MouseUp += Frame_MouseUp;
            Frame.LostMouseCapture += (s, e) => { dragButton = null; };

            KeyDown += MainWindow_KeyDown;

            renderThread = new Thread(renderThreadMain);
            renderThread.Start();
        }

        // Arrow keys move the light in the XZ plane, PageUp/PageDown move it
        // vertically - lets you reposition the (heuristically-placed) light at
        // runtime if FitLightToModel's default guess ends up occluded. The window
        // title is updated with the light's world position so you can read off a
        // value to hardcode once you find one that works.
        private void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            float step = sampleRenderer.LightMoveStep;
            Vec3 delta;
            switch (e.Key)
            {
                case Key.Left: delta = new Vec3(-step, 0, 0); break;
                case Key.Right: delta = new Vec3(step, 0, 0); break;
                case Key.Up: delta = new Vec3(0, 0, -step); break;
                case Key.Down: delta = new Vec3(0, 0, step); break;
                case Key.PageUp: delta = new Vec3(0, step, 0); break;
                case Key.PageDown: delta = new Vec3(0, -step, 0); break;
                default: return;
            }

            sampleRenderer.LightOrigin += delta;
            Title = $"ILGPU.Optix Sample10 - LightOrigin = ({sampleRenderer.LightOrigin.x:0.0}, {sampleRenderer.LightOrigin.y:0.0}, {sampleRenderer.LightOrigin.z:0.0})";
        }

        private void Frame_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton != MouseButton.Left &&
                e.ChangedButton != MouseButton.Right &&
                e.ChangedButton != MouseButton.Middle)
            {
                return;
            }

            dragButton = e.ChangedButton;
            dragLastPos = e.GetPosition(Frame);
            Frame.CaptureMouse();
        }

        private void Frame_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragButton == null || dragLastPos == null)
            {
                return;
            }

            Point pos = e.GetPosition(Frame);
            float dx = (float)((pos.X - dragLastPos.Value.X) / Frame.ActualWidth);
            float dy = (float)((pos.Y - dragLastPos.Value.Y) / Frame.ActualHeight);
            dragLastPos = pos;

            Camera current = sampleRenderer.camera;
            Camera updated = dragButton switch
            {
                MouseButton.Left => CameraMotion.Orbit(current, dx, dy),
                MouseButton.Right => CameraMotion.Dolly(current, dy),
                MouseButton.Middle => CameraMotion.Pan(current, dx, dy),
                _ => current
            };
            sampleRenderer.setCamera(updated);
        }

        private void Frame_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == dragButton)
            {
                dragButton = null;
                Frame.ReleaseMouseCapture();
            }
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            run = false;
            sampleRenderer.Dispose();
        }

        public void renderThreadMain()
        {
            while (run)
            {
                sampleRenderer.render();
            }
        }

        public void draw(ref byte[] data)
        {
            if (data.Length == wBitmap.PixelWidth * wBitmap.PixelHeight * 4)
            {
                wBitmap.Lock();
                IntPtr pBackBuffer = wBitmap.BackBuffer;
                Marshal.Copy(data, 0, pBackBuffer, data.Length);
                wBitmap.AddDirtyRect(rect);
                wBitmap.Unlock();
            }
        }
    }
}
