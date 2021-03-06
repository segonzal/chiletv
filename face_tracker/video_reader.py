import time
import cv2
from queue import Queue
from typing import Tuple, Callable, Union
from threading import Thread
from pathlib import Path


class VideoReader:
    def __init__(self, frame_rate: float, transform: Callable = None, maxsize: int = 128):
        self.frame_rate = frame_rate
        self.transform = transform
        self.stream = cv2.VideoCapture()
        self.frame_queue = Queue(maxsize=maxsize)
        self.stopped = False
        self.thread = None
        self._width = None
        self._height = None
        self._filename = None

    def open(self, filename: Union[str, Path]):
        self._filename = str(filename)
        self.stream.open(self._filename)
        self.get_shape(force=True)

    def close(self):
        if not self.stopped:
            self.stop()
        self.stream.release()

    def clear_queue(self):
        # self.stream.set(cv2.CAP_PROP_POS_MSEC, 0)
        # Empty the queue if there are any elements in it
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            self.frame_queue.all_tasks_done.notify_all()
            self.frame_queue.unfinished_tasks = 0

    def start(self):
        self.stream.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def stop(self):
        self.stopped = True
        self._width = None
        self._height = None
        self.thread.join()

    def read(self):
        frame, timestamp = self.frame_queue.get()
        assert frame is not None, "Frame is None"
        return frame, timestamp

    def update(self):
        ptime = 0
        dtime = 1.0 / self.frame_rate
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.clear_queue()
        while not self.stopped:
            if not self.frame_queue.full():
                ok, frame = self.stream.read()
                stime = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not ok:
                    self.stopped = True
                if dtime - (stime - ptime) < 1e-3:
                    if self.transform:
                        frame = self.transform(frame)
                    self.frame_queue.put((frame, stime))
                    ptime = stime
            else:
                time.sleep(0.1)

    def more(self):
        tries = 0
        while self.frame_queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.frame_queue.qsize() > 0

    def running(self):
        return self.more() or not self.stopped

    def get_shape(self, force: bool = False) -> Tuple[int, int]:
        if force or self._width is None or self._height is None:
            self._width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            self._height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(self._width), int(self._height)

    def get_duration(self) -> float:
        return self.stream.get(cv2.CAP_PROP_FRAME_COUNT) / self.stream.get(cv2.CAP_PROP_FPS)


class BatchedVideoReader(VideoReader):
    def __init__(self, frame_rate: float, batch_size: int = 1, transform: Callable = None, maxsize: int = 128):
        super(BatchedVideoReader, self).__init__(frame_rate, transform, maxsize)
        self.batch_size = batch_size

    def read_batch(self):
        frame_batch = []
        while self.running():
            frame_batch.append(self.read())

            if len(frame_batch) >= self.batch_size:
                yield tuple(zip(*frame_batch))
                frame_batch.clear()

        if len(frame_batch) > 0:
            yield tuple(zip(*frame_batch))

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
