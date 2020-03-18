import numpy as np
import time
import threading

def regular():
    temp = np.random.randn(size * 10, size)
    temp2 = np.random.rand(size, size)
    start = time.time()
    np.dot(temp, temp2)
    print("Regular: {}".format(time.time() - start))

def threaded():
    temp = np.random.randn(size * 10, size)
    print(temp.flags)
    temp2 = np.random.rand(size, size)
    start = time.time()

    thread_list = []
    chunks = 5
    temp_split = np.array(np.split(temp, chunks))

    for i in range(chunks):
        temp_thread = threading.Thread(target=np.dot, args=(temp_split[i], temp2))
        thread_list.append(temp_thread)
        temp_thread.start()
        
    for thread in thread_list:
        thread.join(30)
        
    print("Threaded: {}".format(time.time() - start))
    
size = 1000


for _ in range(1):
    regular()
    threaded()
    print()
