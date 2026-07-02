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

namespace Sample12
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

        // Matches example11_denoiseColorOnly/main.cpp's key bindings: D/space toggles
        // the denoiser, A toggles accumulation/progressive refinement, ,/. adjust
        // samples-per-pixel.
        private void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.D:
                case Key.Space:
                    sampleRenderer.DenoiserOn = !sampleRenderer.DenoiserOn;
                    break;
                case Key.A:
                    sampleRenderer.Accumulate = !sampleRenderer.Accumulate;
                    break;
                case Key.OemComma:
                    sampleRenderer.NumPixelSamples = Math.Max(1, sampleRenderer.NumPixelSamples - 1);
                    break;
                case Key.OemPeriod:
                    sampleRenderer.NumPixelSamples = Math.Max(1, sampleRenderer.NumPixelSamples + 1);
                    break;
                default:
                    return;
            }

            Title = $"ILGPU.Optix Sample12 - Denoiser {(sampleRenderer.DenoiserOn ? "ON" : "OFF")}, " +
                $"Accumulate {(sampleRenderer.Accumulate ? "ON" : "OFF")}, " +
                $"{sampleRenderer.NumPixelSamples} samples/px";
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
