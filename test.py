import threading
import time

# 创建一个Event对象
event = threading.Event()


def my_task():
    while not event.is_set():
        # 执行任务
        print("Task is running")
        time.sleep(1)
    print("Task stopped")


# 创建线程
thread = threading.Thread(target=my_task)
thread.start()

# 等待几秒后停止任务
time.sleep(5)
event.set()  # 设置Event对象，通知线程停止
thread.join()  # 等待线程结束