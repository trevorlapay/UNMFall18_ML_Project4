#%% Imports and constants
import imageio
import glob
import numpy as np
import queue
import threading
import time

NUM_THREADS = 4

existing = glob.glob("test/*green.npy") + glob.glob("train/*green.npy")

#%% Threading
class myThread (threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.id = threadID
        self.q = q
    def run(self):
        print("Starting thread_" + str(self.id)+" for making .npy files.")
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                im_path = self.q.get()
                queueLock.release()
                im = imageio.imread(im_path)
                np.save(im_path[:-4], im)
            else:
                queueLock.release()
                time.sleep(1)
        print("Exiting thread_" + str(self.id))

#%% Script
exitFlag = 0

queueLock = threading.Lock()
workQueue = queue.Queue(100)
threads = []

# Create new threads
for threadID in range(1, NUM_THREADS+1):
   thread = myThread(threadID, workQueue)
   thread.start()
   threads.append(thread)

# Fill the queue
for im_path in glob.glob("test/*green.png") + glob.glob("train/*green.png"):
    if im_path[:-4]+".npy" not in existing:
        while workQueue.full():
            time.sleep(1)
        queueLock.acquire()
        workQueue.put(im_path)
        queueLock.release()

#%% Wrap up
# Wait for queue to empty
while not workQueue.empty():
   time.sleep(1)

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print ("Exiting Main Thread")
