// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: VideoReader.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using OpenCvSharp;

namespace Sample13
{
    // Ported from example/YetAnotherConsoleGameEngine/ConsoleGame/Utils/
    // AsyncFFMPEGVideoReader.cs (NullEngine.Video namespace there) - shells out to
    // ffmpeg.exe to decode a video file to a raw BGR24/BGRA frame stream on a
    // background thread, matching the museum scenes' video-textured-box demos.
    // Two deliberate deviations from the reference, both required for this project's
    // net8.0-windows target (the reference is net471-only):
    //  - `nint` (C# 9 native-int keyword) replaced with `IntPtr` everywhere - Sample13
    //    is pinned to LangVersion 8.0.
    //  - Dispose() kills the ffmpeg process (and closes its stdout pipe) BEFORE
    //    Join()-ing the read thread, instead of after. The reference's original order
    //    (Join(1000) then Thread.Abort() if still alive) relies on Thread.Abort as a
    //    fallback for a read thread stuck in a blocking Stream.Read on ffmpeg's stdout -
    //    but Thread.Abort throws PlatformNotSupportedException on .NET Core/.NET 5+.
    //    Killing the process first makes the blocking read unblock (with a 0-byte
    //    read) on its own well within the join timeout, so the abort fallback is never
    //    needed.
    internal static class WindowsJob
    {
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(IntPtr hJob, JOBOBJECTINFOCLASS infoClass,
            IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

        private enum JOBOBJECTINFOCLASS
        {
            BasicLimitInformation = 2,
            ExtendedLimitInformation = 9,
        }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public IntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private static readonly IntPtr jobHandle;

        static WindowsJob()
        {
            jobHandle = CreateJobObject(IntPtr.Zero, null);

            var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            int length = Marshal.SizeOf(info);
            IntPtr infoPtr = Marshal.AllocHGlobal(length);
            Marshal.StructureToPtr(info, infoPtr, false);

            SetInformationJobObject(jobHandle, JOBOBJECTINFOCLASS.ExtendedLimitInformation,
                infoPtr, (uint)length);

            Marshal.FreeHGlobal(infoPtr);
        }

        public static void AddProcess(Process process)
        {
            if (process != null && !process.HasExited)
            {
                AssignProcessToJobObject(jobHandle, process.Handle);
            }
        }
    }

    internal sealed class AsyncFfmpegVideoReader : IDisposable
    {
        private Thread frameReadThread;
        private bool isRunning;
        private volatile bool hasLooped;

        private readonly object bufferLock = new object();
        private readonly IntPtr[] frameBuffers = new IntPtr[2];
        private int currentBufferIndex = 0;

        private Process ffmpegProcess;
        private Stream ffmpegStdOut;
        private byte[] readBuffer;

        public string VideoFile { get; }
        public int Width { get; }
        public int Height { get; }
        public double Fps { get; }

        private readonly int bytesPerPixel;
        private readonly int frameBytes;

        private Stopwatch timer;

        public bool HasLooped => hasLooped;

        public AsyncFfmpegVideoReader(string videoFile, bool useRGBA = false)
        {
            VideoFile = videoFile;

            // Use OpenCV only to extract metadata.
            using (var tmpCap = new VideoCapture(videoFile, VideoCaptureAPIs.FFMPEG))
            {
                if (!tmpCap.IsOpened())
                    throw new ArgumentException($"Could not open video file: {videoFile}");
                Width = tmpCap.FrameWidth;
                Height = tmpCap.FrameHeight;
                Fps = tmpCap.Fps;
            }

            string ffmpegPixFmt = useRGBA ? "bgra" : "bgr24";
            bytesPerPixel = useRGBA ? 4 : 3;
            frameBytes = Width * Height * bytesPerPixel;

            frameBuffers[0] = Marshal.AllocHGlobal(frameBytes);
            frameBuffers[1] = Marshal.AllocHGlobal(frameBytes);

            ffmpegProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ffmpeg.exe",
                    Arguments = $"-hwaccel cuda -i \"{videoFile}\" -f rawvideo -pix_fmt {ffmpegPixFmt} pipe:1",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            ffmpegProcess.ErrorDataReceived += (sender, e) => { /* Optional logging */ };
            ffmpegProcess.Start();
            WindowsJob.AddProcess(ffmpegProcess);
            ffmpegProcess.BeginErrorReadLine();
            ffmpegStdOut = ffmpegProcess.StandardOutput.BaseStream;

            readBuffer = new byte[frameBytes];

            isRunning = true;
            frameReadThread = new Thread(FrameReadLoop) { IsBackground = true };
            frameReadThread.Start();

            timer = Stopwatch.StartNew();
        }

        private void FrameReadLoop()
        {
            double nextFrameTimestamp = timer.Elapsed.TotalMilliseconds;
            double frameDuration = 1000.0 / Fps;
            while (isRunning)
            {
                double currentMs = timer.Elapsed.TotalMilliseconds;
                if (currentMs < nextFrameTimestamp)
                {
                    double remaining = nextFrameTimestamp - currentMs;
                    if (remaining > 2.0)
                    {
                        Thread.Sleep((int)(remaining - 1));
                    }
                    else
                    {
                        Thread.SpinWait(50);
                    }
                    continue;
                }
                if (!ReadOneFrame(out bool looped))
                {
                    LoopOrBreak();
                }
                else if (looped)
                {
                    hasLooped = true;
                }
                nextFrameTimestamp += frameDuration;
            }
        }

        private bool ReadOneFrame(out bool looped)
        {
            looped = false;
            int totalRead = 0;
            while (totalRead < frameBytes)
            {
                int n = ffmpegStdOut.Read(readBuffer, totalRead, frameBytes - totalRead);
                if (n <= 0)
                {
                    return false;
                }
                totalRead += n;
            }
            int nextBufferIndex = 1 - currentBufferIndex;
            Marshal.Copy(readBuffer, 0, frameBuffers[nextBufferIndex], frameBytes);
            lock (bufferLock)
            {
                currentBufferIndex = nextBufferIndex;
            }
            return totalRead == frameBytes;
        }

        private void LoopOrBreak()
        {
            if (!isRunning)
                return;

            try
            {
                ffmpegProcess?.Kill();
            }
            catch
            {
            }

            ffmpegProcess?.Dispose();
            hasLooped = true;
            var info = ffmpegProcess?.StartInfo;
            if (info == null) return;
            ffmpegProcess = new Process { StartInfo = info };
            ffmpegProcess.ErrorDataReceived += (sender, e) => { };
            ffmpegProcess.Start();
            WindowsJob.AddProcess(ffmpegProcess);
            ffmpegProcess.BeginErrorReadLine();
            ffmpegStdOut = ffmpegProcess.StandardOutput.BaseStream;
        }

        public IntPtr GetCurrentFramePtr()
        {
            lock (bufferLock)
            {
                return frameBuffers[currentBufferIndex];
            }
        }

        public void Dispose()
        {
            isRunning = false;

            // Kill ffmpeg and close its stdout pipe first, so the read thread's
            // blocking Stream.Read unblocks (0-byte read) well within the join
            // timeout - see the class-level comment on why this ordering matters.
            try
            {
                ffmpegStdOut?.Close();
                if (ffmpegProcess != null && !ffmpegProcess.HasExited)
                {
                    ffmpegProcess.Kill();
                    ffmpegProcess.WaitForExit();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Dispose(): Exception on ffmpeg shutdown: " + ex);
            }
            ffmpegProcess?.Dispose();

            if (frameReadThread != null && frameReadThread.IsAlive)
                frameReadThread.Join(2000);

            for (int i = 0; i < frameBuffers.Length; i++)
            {
                if (frameBuffers[i] != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(frameBuffers[i]);
                    frameBuffers[i] = IntPtr.Zero;
                }
            }
        }
    }
}
