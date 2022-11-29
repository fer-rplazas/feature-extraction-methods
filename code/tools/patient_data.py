import numpy as np
from enum import Enum
import h5py
from pathlib import Path

PatID = Enum("PatIDs", ["ET1", "ET2", "ET3", "ET4", "ET5", "ET6", "ET7", "ET8"])
Task = Enum("Task", ["Pegboard", "Pouring", "Posture"])
Stim = Enum("Stim", ["ON", "OFF"])

DATA_PATH = Path(__file__).parent / Path("../../patient_data/")


class StimInit(type):
    def __class_getitem__(cls, name: str | Stim):
        if isinstance(name, str):
            return Stim[name.upper()]
        elif isinstance(name, Stim):
            return name
        else:
            raise KeyError("Stim not recognized")


class PatDataset:
    def __init__(self, patID: PatID, task: Task, stim: Stim):

        keys = []
        for el, enum_type in zip([patID, task, stim], [PatID, Task, StimInit]):
            try:
                keys.append(enum_type[el] if not isinstance(el, enum_type) else el)
            except KeyError:
                raise ValueError(f"""`{el}` not recognized!""")

        self.patID, self.task, self.stim = keys[0], keys[1], keys[2]

        self.fpath = (
            DATA_PATH
            / str(self.patID.name)
            / f"{self.task.name}_{self.stim.name.lower()}.h5"
        )

        if self.fpath.exists():
            self.exists_ = True

        else:
            self.exists_ = False

        self.signals, self.label = None, None

    def load(self):
        if self.exists():
            f = h5py.File(self.fpath, "r")
            self.signals = f["LFP"][:]
            self.label = f["label"][:]
            self.fs = 2048.0  # Hardcoded since all datasets here are sampled at 2048 Hz
        else:
            self.signals, self.label, self.fs = None, None, None

        return self

    def exists(self):
        return self.exists_


if __name__ == "__main__":
    data = PatDataset("ET1", "Pegboard", "Off").load()
