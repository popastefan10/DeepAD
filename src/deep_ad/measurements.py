import subprocess as sp
import time


class Stopwatch:
    """Measures time in seconds."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.last_checkpoint: float | None = None

    def start(self) -> float:
        """Starts the stopwatch and returns the current time in seconds."""
        self.start_time = time.time()
        self.last_checkpoint = self.start_time

        return self.start_time

    def checkpoint(self, return_delta: bool = True) -> float:
        """
        Saves current time as a checkpoint and returns the time elapsed since the last checkpoint if `return_delta=True`,
        otherwise the current time.
        """
        prev_checkpoint = self.last_checkpoint
        self.last_checkpoint = time.time()

        return (self.last_checkpoint - prev_checkpoint) if return_delta else self.last_checkpoint

    def elapsed_since_beginning(self) -> float:
        """Returns the total time elapsed in seconds since it was started."""
        return time.time() - self.start_time

    def elapsed_since_last_checkpoint(self, last_checkpoint: float | None = None) -> float:
        """Returns the time elapsed in seconds since the last checkpoint, without saving a new checkpoint."""
        return time.time() - (last_checkpoint or self.last_checkpoint)


# https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
def get_gpu_memory() -> list[int]:
    """Returns the free memory in MiB for each GPU on the system."""
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    return memory_free_values


class GPUWatch:
    """Measures free memory for a specific GPU in MiB."""

    @staticmethod
    def current_free_memory(gpu_index: int = 0) -> int:
        """Returns the current free memory in MiB for the specified GPU."""
        return get_gpu_memory()[gpu_index]

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self.start_memory: int | None = None
        self.last_checkpoint: int | None = None

    def start(self) -> int:
        """Starts the GPUWatch and returns the current free memory in MiB."""
        self.start_memory = get_gpu_memory()[self.gpu_index]
        self.last_checkpoint = self.start_memory

        return self.start_memory

    def checkpoint(self, return_delta: bool = True) -> int:
        """
        Saves current memory as a checkpoint and returns the memory gain since the last checkpoint.

        Args:
            * `return_delta`: If True, returns the memory gain since the last checkpoint. Otherwise, returns the current free GPU memory.
            * `last_checkpoint`: The last checkpoint to compare against when computing delta, if `return_delta=True`.
        """
        prev_checkpoint = self.last_checkpoint
        self.last_checkpoint = get_gpu_memory()[self.gpu_index]

        return (self.last_checkpoint - prev_checkpoint) if return_delta else self.last_checkpoint

    def gain_since_beginning(self) -> int:
        """Returns the total memory gain in MiB since it was started."""
        return get_gpu_memory()[self.gpu_index] - self.start_memory

    def gain_since_last_checkpoint(self, last_checkpoint: int | None = None) -> int:
        """Returns the memory gain in MiB since the last checkpoint, without saving a new checkpoint."""
        return get_gpu_memory()[self.gpu_index] - (last_checkpoint or self.last_checkpoint)
