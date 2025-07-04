# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainers."""

from .base import Trainer
from .classification import GoalFrameClassifierTrainer
from .lifs import LIFSTrainer
from .tcc import TCCTrainer
from .tcn import TCNCrossEntropyTrainer
from .tcn import TCNTrainer
from .holdr import HOLDRTrainer
from .reds import REDSRewardTrainer

__all__ = [
    "Trainer",
    "TCCTrainer",
    "TCNTrainer",
    "TCNCrossEntropyTrainer",
    "LIFSTrainer",
    "GoalFrameClassifierTrainer",
    "HOLDRTrainer", 
    "REDSRewardTrainer"
]
