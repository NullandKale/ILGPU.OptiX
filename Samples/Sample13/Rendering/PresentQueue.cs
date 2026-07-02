using System;
using System.Threading;

namespace Sample13
{
    /// <summary>
    /// Double-buffered async present handoff between the render thread and the UI
    /// thread, DX12-swapchain style (2 buffers, max frame latency 2).
    ///
    /// Render and present interleave: the render thread can start producing the next
    /// frame into the other buffer as soon as it publishes one, while the UI thread is
    /// still presenting the previous one, but the render thread blocks
    /// (bufferFree[idx].Wait, see TryBeginWrite) once it's produced 2 frames the UI
    /// thread hasn't consumed yet. That backpressure is what couples the two rates
    /// together - unlike a fully decoupled "always grab the latest" scheme, frames are
    /// presented strictly in production order and the render thread's own pace is
    /// throttled by how fast the UI thread actually consumes them, the same way a real
    /// swap chain's CPU submission thread stalls on its frame-latency-waitable object.
    /// Each buffer index has its own pair of semaphores: bufferFree[i] (1,1) starts
    /// signaled - "render may write here" - and bufferReady[i] (0,1) - "UI may read
    /// this, render just published it". Only ever touched via SemaphoreSlim.Wait/
    /// Release, never a plain lock.
    ///
    /// IMPORTANT: this is deliberately the ONLY renderer component that is safe to (and
    /// must be) used OUTSIDE the renderer's gpuLock - see the lock's comment in
    /// SampleRenderer.cs for the deadlock this avoids.
    /// </summary>
    public sealed class PresentQueue : IDisposable
    {
        readonly byte[][] frameBuffers;
        readonly SemaphoreSlim[] bufferFree;
        readonly SemaphoreSlim[] bufferReady;
        int writeIndex;

        // Index 0 is MainWindow's initial read index (see TryPresentFrame and
        // MainWindow's presentIndex field) and also the initial writeIndex - fine,
        // since the render thread always claims bufferFree[writeIndex] before writing,
        // and both start signaled-free.
        public PresentQueue(int frameBytes)
        {
            frameBuffers = new[] { new byte[frameBytes], new byte[frameBytes] };
            bufferFree = new[] { new SemaphoreSlim(1, 1), new SemaphoreSlim(1, 1) };
            bufferReady = new[] { new SemaphoreSlim(0, 1), new SemaphoreSlim(0, 1) };
            writeIndex = 0;
        }

        // Render-thread side: claims the current write slot, handing back the CPU
        // buffer to fill. Waits until the UI thread is done reading this slot from two
        // frames ago (see the class comment) - in the overwhelmingly common case this
        // is immediate, since the UI's single Marshal.Copy is far cheaper than a GPU
        // trace/denoise/tonemap pass; it only actually blocks if presentation has
        // fallen more than one frame behind, which is exactly the DX12-style
        // backpressure that couples the two rates together. Bounded retries (rather
        // than one blocking Wait) keep shutdown responsive if the UI stops pumping
        // CompositionTarget.Rendering (window closing) before draining this slot -
        // keepWaiting is polled between retries and false aborts the write.
        public bool TryBeginWrite(Func<bool> keepWaiting, out byte[] target)
        {
            for (; ; )
            {
                if (bufferFree[writeIndex].Wait(50))
                {
                    target = frameBuffers[writeIndex];
                    return true;
                }
                if (!keepWaiting())
                {
                    target = null;
                    return false;
                }
            }
        }

        // Render-thread side: publishes the slot TryBeginWrite handed out and advances
        // to the other buffer.
        public void EndWrite()
        {
            bufferReady[writeIndex].Release();
            writeIndex = 1 - writeIndex;
        }

        // Reader side of the handoff - called once per WPF-composed frame from
        // MainWindow's CompositionTarget.Rendering handler. readIndex is the caller's
        // private "next buffer to read" index (starts at 0); passing it by ref lets
        // this update it in place. Non-blocking: if the render thread hasn't published
        // a fresh frame into this slot yet, returns false and the caller just leaves
        // whatever's already on screen (WPF retains the WriteableBitmap's last contents
        // automatically). Frames are always consumed in the exact order the render
        // thread produced them (0,1,0,1,...), never "whichever's newest" - that
        // in-order handoff plus TryBeginWrite's backpressure is what couples the two
        // threads' rates together, unlike a fully decoupled latest-wins scheme.
        public bool TryPresentFrame(ref int readIndex, out byte[] data)
        {
            if (!bufferReady[readIndex].Wait(0))
            {
                data = null;
                return false;
            }

            data = frameBuffers[readIndex];
            return true;
        }

        // Called once the caller (MainWindow.draw) has finished copying the buffer
        // TryPresentFrame just handed back - frees the slot for the render thread to
        // reuse two frames from now, and advances readIndex to the next slot.
        public void FinishPresent(ref int readIndex)
        {
            bufferFree[readIndex].Release();
            readIndex = 1 - readIndex;
        }

        public void Dispose()
        {
            bufferFree[0].Dispose();
            bufferFree[1].Dispose();
            bufferReady[0].Dispose();
            bufferReady[1].Dispose();
        }
    }
}
