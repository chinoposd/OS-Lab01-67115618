import threading
import time

# Simulating Hardware Resources (OS Mutex Locks)
gpu_0_lock = threading.Lock()
gpu_1_lock = threading.Lock()


def train_model_a():
    """Model A needs GPU 0 first, then GPU 1"""

    print("[Model A] Waiting to acquire GPU 0...")

    gpu_0_lock.acquire()

    print("[Model A] Successfully acquired GPU 0! Processing...")

    # Force OS context switch
    time.sleep(0.1)

    print("[Model A] Waiting to acquire GPU 1...")

    gpu_1_lock.acquire()

    print("[Model A] Successfully acquired GPU 1! Training complete.")

    gpu_1_lock.release()
    gpu_0_lock.release()


def train_model_b():
    """Model B needs GPU 1 first, then GPU 0"""

    print("[Model B] Waiting to acquire GPU 1...")

    gpu_1_lock.acquire()

    print("[Model B] Successfully acquired GPU 1! Processing...")

    # Force OS context switch
    time.sleep(0.1)

    print("[Model B] Waiting to acquire GPU 0...")

    gpu_0_lock.acquire()

    print("[Model B] Successfully acquired GPU 0! Training complete.")

    gpu_0_lock.release()
    gpu_1_lock.release()


def main():
    print("--- Starting ML Training Cluster ---")

    t1 = threading.Thread(target=train_model_a)
    t2 = threading.Thread(target=train_model_b)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("--- Cluster Execution Completed ---")


if __name__ == "__main__":
    main()