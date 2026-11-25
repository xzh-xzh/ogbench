# Labeled PointMaze Navigate

This dataset contains newly collected trajectories for the Medium, Large and Giant PointMaze maps. It uses the OGBench Navigate collection protocol and adds the behavior goal active at each action, the goal-resampling event, and the boundaries of each behavior-goal segment. The original OGBench datasets remain available from upstream. [OGBench project](https://seohong.me/projects/ogbench/) · [Collection source at the upstream revision](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/data_gen_scripts/generate_locomaze.py)

## Download

Each map has an independent training and validation file. Each split contains 1,000,000 training or 100,000 validation transitions under the OGBench convention of excluding the final row of each record from the action samples. The six archives total 91.6 MB. [PointMaze Labeled v1.0.0 release](https://github.com/xzh-xzh/ogbench/releases/tag/pointmaze-labeled-v1.0.0)

| Map | Training | Validation | Saved rows (train / val) | Records (train / val) |
| --- | --- | --- | ---: | ---: |
| Medium | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-medium-navigate-v0.npz) | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-medium-navigate-v0-val.npz) | 1,001,000 / 100,100 | 1,000 / 100 |
| Large | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-large-navigate-v0.npz) | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-large-navigate-v0-val.npz) | 1,001,000 / 100,100 | 1,000 / 100 |
| Giant | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-giant-navigate-v0.npz) | [NPZ](https://github.com/xzh-xzh/ogbench/releases/download/pointmaze-labeled-v1.0.0/pointmaze-giant-navigate-v0-val.npz) | 1,000,500 / 100,050 | 500 / 50 |

Keep each pair together, with the original filenames, in a dedicated directory such as `data/pointmaze-labeled/`. The filenames follow the upstream convention so that the OGBench API can find the validation file. The manifest gives file sizes, SHA-256 checksums, fields, and collection settings for each archive. [Dataset manifest](manifest.json)

```sh
mkdir -p data/pointmaze-labeled
gh release download pointmaze-labeled-v1.0.0 --repo xzh-xzh/ogbench \
  --pattern '*.npz' --pattern 'SHA256SUMS.txt' --dir data/pointmaze-labeled
(cd data/pointmaze-labeled && sha256sum -c SHA256SUMS.txt)
```

## Fields and alignment

| Fields | Shape and dtype | Meaning |
| --- | --- | --- |
| `observations`, `next_observations`, `actions`, `qpos`, `qvel` | `(N, 2)` `float32` | Current and post-step observations, action, and pre-step MuJoCo position and velocity |
| `behavior_goals`, `next_behavior_goals` | `(N, 2)` `float32` | Target active before action `t`, and target after the step and any immediate resampling |
| `goal_success`, `goal_switch` | `(N,)` `bool` | Native success and the ensuing target-resampling event. A resample can select the same target again |
| `terminals` | `(N,)` `bool` | Marks the final saved row of each record |
| `record_id` | `(N,)` `int64` | Continuous collection record ID. Validation may start at a nonzero ID |
| `behavior_segment_id`, `segment_start`, `segment_end` | `(N,)` `int64` | Segment ID, inclusive first action index, and exclusive action boundary. Row indices are local to the split |
| `valids` (Medium only) | `(N,)` `float32` | `1` for a real action and `0` for the final dummy row |

- **Medium** records have 1,000 real actions followed by a final observation row. Its `valids` is `0` and `behavior_segment_id` is `-1`.
- **Large and Giant** records retain 1,001 and 2,001 real-action rows, respectively, including the time-limit action with its explicit `next_observations`. For the standard shifted-state convention, exclude terminal actions with `~terminals`. Before using `segment_end` to index `observations`, cap it at the final saved row of its own record.
- For every nonterminal action `t`, `next_observations[t]` equals `observations[t + 1]`. Preserve record boundaries when forming shifted transitions or sampling future states.
- The position at a goal switch is `next_observations[t]` where `goal_switch[t]` is true. The labels cover changes in the collection target and segment boundaries. Geometric bends and intermediate oracle waypoints have no dedicated labels.

## Load the data

Use NumPy to access the recorded goals and segment labels directly. This example extracts transitions using the standard OGBench row convention, alongside the observed positions at all recorded goal switches.

```python
from pathlib import Path

import numpy as np

path = Path("data/pointmaze-labeled/pointmaze-medium-navigate-v0.npz")
with np.load(path, allow_pickle=False) as data:
    keep = ~data["terminals"]
    states = data["observations"][keep]
    actions = data["actions"][keep]
    next_states = data["next_observations"][keep]
    goals = data["behavior_goals"][keep]
    segment_ids = data["behavior_segment_id"][keep]
    switch_xy = data["next_observations"][data["goal_switch"]]
```

Use the upstream API to load the standard OGBench training fields and create the corresponding evaluation environment. Read the extra labels with the NumPy example above.

```python
import ogbench

env, train, val = ogbench.make_env_and_datasets(
    "pointmaze-medium-navigate-v0",
    dataset_path="data/pointmaze-labeled/pointmaze-medium-navigate-v0.npz",
)
```

## Collection protocol

The collector samples the start from free maze cells and each target from free cells outside straight corridors. A scripted PointMaze oracle points toward its next routing subgoal. Independent Gaussian noise with standard deviation `0.5` is added to the action components before clipping to `[-1, 1]`. When the environment reports success, the collector resamples the behavior target and continues the same record. The labels preserve the target immediately before the action and immediately after the resulting step and resample. [Upstream Navigate settings](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/data_gen_scripts/commands.sh)

- Medium used deterministic per-record seeds (`470000 + record index` for train, `570000 + record index` for validation), NumPy 1.26.4, Gymnasium 1.2.3, and MuJoCo 3.3.6.
- Large and Giant retain the upstream collector's sequential train and 10% validation records and its unseeded PointMaze random stream. They used NumPy 2.3.4, Gymnasium 1.2.2, and MuJoCo 3.10.0. Use the released archives and their checksums for exact data reuse.

The OGBench environments, collection design, and reference implementation come from the OGBench authors. [OGBench paper](https://arxiv.org/abs/2410.20092) · [Upstream software license](https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/LICENSE)

## 中文说明

这套数据用 OGBench PointMaze 的 Navigate 采集流程重新跑了 Medium、Large 和 Giant。每个地图有训练、验证两份 NPZ。按 OGBench 口径，每个地图有 100 万条训练转移和 10 万条验证转移。上面的下载表可以直接逐个下载。

`behavior_goals[t]` 是执行第 `t` 个动作时追踪的目标。环境判定到达后，采集器会重采样目标并打上 `goal_switch[t]`，然后在同一条记录中继续移动。切换发生的位置可从 `next_observations[t]` 读取。`segment_start` 和 `segment_end` 给出同一目标段的动作范围，左闭右开。这里的「转折点」对应行为目标切换，数据没有单独标注墙角或 oracle 中间路点。

读取目标和分段字段时，直接用 `numpy.load(..., allow_pickle=False)`。按标准 OGBench 口径训练时用 `~terminals` 取动作行。Medium 末行只保存最后的状态，Large 和 Giant 的末行还保存了到达时限的真实动作。用 `segment_end` 索引 Large 或 Giant 的 `observations` 时，应截到同一 `record_id` 的最后一行。
