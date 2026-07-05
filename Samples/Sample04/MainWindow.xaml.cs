// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: MainWindow.xaml.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Sample04
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

            wBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            rect = new Int32Rect(0, 0, width, height);
            Frame.Source = wBitmap;

            sampleRenderer = new SampleRenderer(width, height, this);
            Closing += MainWindow_Closing;

            Frame.MouseDown += Frame_MouseDown;
            Frame.MouseMove += Frame_MouseMove;
            Frame.MouseUp += Frame_MouseUp;
            Frame.LostMouseCapture += (s, e) => { dragButton = null; };

            renderThread = new Thread(renderThreadMain);
            renderThread.Start();
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
