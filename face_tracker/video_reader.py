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

    def start(self, filename: Union[str, Path]):
        if isinstance(filename, Path):
            filename = str(filename)
        self.stream.open(filename)
        self.stream.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.stopped = False

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def stop(self):
        self.stopped = True
        self.thread.join()

    def read(self):
        frame, timestamp = self.frame_queue.get()
        assert frame is not None, "Frame is None"
        return frame, timestamp

    def update(self):
        ptime = 0
        dtime = 1.0 / self.frame_rate
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
        self.stream.release()

    def more(self):
        tries = 0
        while self.frame_queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.frame_queue.qsize() > 0

    def running(self):
        return self.more() or not self.stopped

    def get_shape(self) -> Tuple[int, int]:
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def get_duration(self) -> float:
        return self.stream.get(cv2.CAP_PROP_FRAME_COUNT) / self.stream.get(cv2.CAP_PROP_FPS)


class AdaptableBatchedVideoReader(VideoReader):
    def __init__(self, frame_rate: float, batch_size: int, transform: Callable = None, maxsize: int = 128):
        super(AdaptableBatchedVideoReader, self).__init__(frame_rate, transform, maxsize)
        self._batch_size = batch_size

    def read_batch(self):
        frame_batch = []
        while self.running():
            frame_batch.append(self.read())

            if len(frame_batch) >= self.batch_size:
                yield tuple(zip(*frame_batch))
                frame_batch.clear()

        if len(frame_batch) > 0:
            yield tuple(zip(*frame_batch))

    @property
    def batch_size(self) -> int:
        # _batch_size is for images of 32x32
        # returned batch size is for the actual images
        width, height = self.get_shape()
        batch_size = int((width * height * self._batch_size) / (512 * 512))
        # batch_size = min(batch_size, 256)
        return batch_size


class BatchedVideoReader(AdaptableBatchedVideoReader):
    @property
    def batch_size(self) -> int:
        return self._batch_size
