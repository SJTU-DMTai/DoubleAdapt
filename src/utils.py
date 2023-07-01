import bisect
import datetime
import traceback
import warnings
from collections import defaultdict
from typing import Union, List, Tuple, Dict, Text

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


class TimeAdjuster:
    """
    Find appropriate date and adjust date.
    """
    def __init__(self, calendar=Union[list, pd.Series]):
        if isinstance(calendar, list):
            calendar = pd.Series(data=calendar)
        self.cals = pd.to_datetime(calendar)

    def get(self, idx: int):
        """
        Get datetime by index.

        Parameters
        ----------
        idx : int
            index of the calendar
        """
        if idx is None or idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self) -> pd.Timestamp:
        """
        Return the max calendar datetime
        """
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start") -> int:
        """
        Align the index of time_point in the calendar.

        Parameters
        ----------
        time_point
        tp_type : str

        Returns
        -------
        index : int
        """
        if time_point is None:
            # `None` indicates unbounded index/boarder
            return None
        time_point = pd.Timestamp(time_point)
        if tp_type == "start":
            idx = bisect.bisect_left(self.cals, time_point)
        elif tp_type == "end":
            idx = bisect.bisect_right(self.cals, time_point) - 1
        else:
            raise NotImplementedError(f"This type of input is not supported")
        return idx

    def align_time(self, time_point, tp_type="start") -> pd.Timestamp:
        """
        Align time_point to trade date of calendar

        Args:
            time_point
                Time point
            tp_type : str
                time point type (`"start"`, `"end"`)

        Returns:
            pd.Timestamp
        """
        if time_point is None:
            return None
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment: Union[dict, tuple]) -> Union[dict, tuple]:
        """
        Align the given date to the trade date

        for example:

            .. code-block:: python

                input: {'train': ('2008-01-01', '2014-12-31'), 'valid': ('2015-01-01', '2016-12-31'), 'test': ('2017-01-01', '2020-08-01')}

                output: {'train': (Timestamp('2008-01-02 00:00:00'), Timestamp('2014-12-31 00:00:00')),
                        'valid': (Timestamp('2015-01-05 00:00:00'), Timestamp('2016-12-30 00:00:00')),
                        'test': (Timestamp('2017-01-03 00:00:00'), Timestamp('2020-07-31 00:00:00'))}

        Parameters
        ----------
        segment

        Returns
        -------
        Union[dict, tuple]: the start and end trade date (pd.Timestamp) between the given start and end date.
        """
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, (tuple, list)):
            return self.align_time(segment[0], tp_type="start"), self.align_time(segment[1], tp_type="end")
        else:
            raise NotImplementedError(f"This type of input is not supported")

    SHIFT_SD = "sliding"
    SHIFT_EX = "expanding"

    def _add_step(self, index, step):
        if index is None:
            return None
        return index + step

    def shift(self, seg: tuple, step: int, rtype=SHIFT_SD) -> tuple:
        """
        Shift the datatime of segment

        If there are None (which indicates unbounded index) in the segment, this method will return None.

        Parameters
        ----------
        seg :
            datetime segment
        step : int
            rolling step
        rtype : str
            rolling type ("sliding" or "expanding")

        Returns
        --------
        tuple: new segment

        Raises
        ------
        KeyError:
            shift will raise error if the index(both start and end) is out of self.cal
        """
        if isinstance(seg, tuple):
            start_idx, end_idx = self.align_idx(seg[0], tp_type="start"), self.align_idx(seg[1], tp_type="end")
            if rtype == self.SHIFT_SD:
                start_idx = self._add_step(start_idx, step)
                end_idx = self._add_step(end_idx, step)
            elif rtype == self.SHIFT_EX:
                end_idx = self._add_step(end_idx, step)
            else:
                raise NotImplementedError(f"This type of input is not supported")
            if start_idx is not None and start_idx > len(self.cals):
                raise KeyError("The segment is out of valid calendar")
            return self.get(start_idx), self.get(end_idx)
        else:
            raise NotImplementedError(f"This type of input is not supported")


def organize_all_tasks(segments: Dict[Text, Tuple], ta: TimeAdjuster,
                       step: int, trunc_days: int = 2,
                       rtype: str = TimeAdjuster.SHIFT_SD,
                       use_extra: bool = False) -> Dict[Text, List[Dict[Text, Tuple]]]:
    """
    Organize training data, validation data, and test data into rolling tasks.

    Parameters
    ----------
    segments (Dict[Text, Tuple]):
        The date range of training data, validation data, and test data.
        Example:
            {
                'train': ('2008-01-01', '2014-12-31'),
                'valid': ('2015-01-01', '2016-12-31'),
                'test': ('2017-01-01', '2020-12-31')
            }
    step (int):
        the rolling interval
    trunc_days (int):
        truncate out the last part of training data to avoid information leakage
    rtype (str):
        if TimeAdjuster.SHIFT_SD, using sliding windows;
        if TimeAdjuster.SHIFT_EX, using all observed data for training.
    use_extra (bool):
        whether to use the extra segment between the support set and the query set for meta-learning optimization.
    """
    all_rolling_tasks = organize_tasks(segments['train'][0], segments['test'][-1], ta, step, trunc_days, rtype, use_extra)
    rolling_tasks = {}
    rolling_tasks['train'], rolling_tasks['test'] = split_rolling_tasks(all_rolling_tasks,
                                                                        split_point=segments['valid'][-1])
    rolling_tasks['train'], rolling_tasks['valid'] = split_rolling_tasks(rolling_tasks['train'],
                                                                         split_point=segments['train'][-1])
    return rolling_tasks


def split_rolling_tasks(rolling_tasks: List[Dict[Text, Tuple]], split_point: Union[str, pd.Timestamp]):
    assert len(rolling_tasks) > 0
    for i, t in enumerate(rolling_tasks):
        if t["test"][-1] >= pd.Timestamp(split_point):
            break
    return rolling_tasks[:i], rolling_tasks[i:]


def organize_tasks(start_date, end_date, ta: TimeAdjuster, step: int, trunc_days=2,
                   rtype=TimeAdjuster.SHIFT_SD, use_extra: bool = False) -> List[Dict[Text, Tuple]]:

    train_begin = start_date
    train_end = ta.get(ta.align_idx(train_begin) + step - 1)
    test_begin = ta.get(ta.align_idx(train_begin) + step - 1 + trunc_days)
    test_end = ta.get(ta.align_idx(test_begin) + step - 1)
    segments = {
        "train": (train_begin, train_end),
        "test": (test_begin, test_end),
    }
    if use_extra:
        extra_begin = ta.get(ta.align_idx(train_end) + 1)
        extra_end = ta.get(ta.align_idx(test_begin) - 1)
        segments["extra"] = (extra_begin, extra_end)

    return generate_rolling_tasks(segments, step, ta, end_date, rtype=rtype)


def generate_rolling_tasks(first_task_segment: Dict[Text, Tuple], step: int,
                           time_adjuster: TimeAdjuster, end_date: Union[datetime.datetime, str],
                           rtype:str) -> List[Dict[Text, Tuple]]:
    rolling_tasks = [first_task_segment]
    while True:
        try:
            task = {}
            for k, v in rolling_tasks[-1].items():
                if k == 'train' and rtype == TimeAdjuster.SHIFT_EX:
                    task[k] = time_adjuster.shift(v, step=step, rtype=rtype)
                else:
                    task[k] = time_adjuster.shift(v, step=step, rtype=TimeAdjuster.SHIFT_SD)
            rolling_tasks.append(task)
            if rolling_tasks[-1]['test'][-1] >= time_adjuster.align_time(end_date, tp_type='end'):
                rolling_tasks[-1]['test'] = (rolling_tasks[-1]['test'][0],
                                             time_adjuster.align_time(end_date, tp_type='end'))
                break
        except:
            traceback.print_exc()
            break
    return rolling_tasks


def get_rolling_data(rolling_task_segments: List[Dict[Text, Tuple]],
                     data: pd.DataFrame,
                     factor_num: int = 6, horizon: int = 1,
                     not_sequence: bool = True, sequence_last_dim: bool = True,
                     to_tensor: bool = True) -> List[Dict[str, Union[np.ndarray, pd.Index, torch.Tensor]]]:
    """
    Fetch dataframes according to a list of segments, which will be preprocessed into features, labels, and dates.

    Parameters
    ----------
    rolling_task_segments (List[Dict[Text, Tuple[datetime.datetime]]]):
        a sequence of task segments
    data (pd.DataFrame):
        the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
        the col named 'feature' contains the stock feature vectors;
        the col named 'label' contains the ground-truth labels.
    factor_num (int):
        # of stock factors (default 6 for Alpha360).
    horizon (int):
        define the stock price trend
    not_sequence (bool):
        whether the stock feature vector is a time series.
    sequence_last_dim (bool):
        whether the time series is aligned along the last axis of dimensions.
    to_tensor (bool):
        if True, transform all `numpy.ndarray` to `torch.Tensor` at once;
        if False, transform each batch from `numpy.ndarray` to `torch.Tensor` (slow, not recommended)
    """
    rolling_tasks_data = []
    for seg in tqdm(rolling_task_segments, desc="creating tasks"):
        rolling_tasks_data.append(get_task_data(seg, data))
    return preprocess(
        rolling_tasks_data,
        factor_num=factor_num,
        sequence_last_dim=sequence_last_dim,
        H=1 + horizon,
        not_sequence=not_sequence,
        to_tensor=to_tensor
    )


def get_task_data(segments: dict, dataframe) -> Dict[str, Union[np.ndarray, pd.Index]]:
    train_exist = "train" in segments
    extra_exist = "extra" in segments
    if train_exist:
        train_segs = [str(dt) for dt in segments["train"]]
    if extra_exist:
        extra_segs = [str(dt) for dt in segments["extra"]]
    test_segs = [str(dt) for dt in segments["test"]]

    d_test = get_data_from_seg(test_segs, dataframe, True)
    processed_meta_input = dict(
        X_test=d_test["feature"], y_test=d_test["label"].iloc[:, 0], test_idx=d_test["label"].index,
    )
    if train_exist:
        d_train = get_data_from_seg(train_segs, dataframe)
        processed_meta_input.update(
            X_train=d_train["feature"], y_train=d_train["label"].iloc[:, 0], train_idx=d_train["label"].index,
        )
    if extra_exist:
        d_extra = get_data_from_seg(extra_segs, dataframe)
        processed_meta_input.update(
            X_extra=d_extra["feature"], y_extra=d_extra["label"].iloc[:, 0], extra_idx=d_extra["label"].index,
        )
    return processed_meta_input


def get_data_from_seg(seg: tuple, dataframe: pd.DataFrame, test: bool=False):
    try:
        d = (
            dataframe.loc(axis=0)[seg[0]: seg[1]]
            if not test or seg[1] <= str(dataframe.index[-1][0])
            else dataframe.loc(axis=0)[seg[0]:]
        )
    except Exception as e:
        traceback.print_exc()
        new_seg = [seg[0], seg[1]]
        all_dates = dataframe.index.levels[0]
        if seg[0] not in all_dates:
            new_seg[0] = all_dates[all_dates > seg[0]][0]
            if str(new_seg[0])[:10] > seg[1]:
                warnings.warn(f"Exceed test time{new_seg}")
                return None
        if seg[1] not in all_dates:
            new_seg[1] = all_dates[all_dates < seg[1]][-1]
            if str(new_seg[1])[:10] < seg[0]:
                warnings.warn(f"Exceed training time{new_seg}")
                return None
            d = (
                dataframe.loc(axis=0)[new_seg[0]: new_seg[1]]
                if not test or new_seg[1] <= all_dates[-1]
                else dataframe.loc(axis=0)[new_seg[0]:]
            )
        else:
            d = (
                dataframe.loc(axis=0)[new_seg[0]: new_seg[1]]
                if not test or new_seg[1] <= str(all_dates[-1])
                else dataframe.loc(axis=0)[new_seg[0]:]
            )
        warnings.warn(f"{seg} becomes {new_seg} after adjustment")
    return d


def preprocess(task_data_list: List[Dict[str, Union[np.ndarray, pd.Index]]],
               factor_num=6, H=1, sequence_last_dim=True, not_sequence=False,
               to_tensor=True,) -> List[Dict[str, Union[np.ndarray, pd.Index, torch.Tensor]]]:
    skip = []
    for i, task_data in enumerate(task_data_list):
        data_type = set()
        for k in task_data.keys():
            if k.startswith("X") or k.startswith("y"):
                data_type.add(k[2:])
                if not isinstance(task_data[k], np.ndarray):
                    task_data[k] = task_data[k].to_numpy()
                if to_tensor:
                    task_data[k] = torch.tensor(task_data[k], dtype=torch.float32)
        if task_data['y_test'].shape[0] == 0:
            skip.append(i)

        if not_sequence:
            for dt in data_type:
                k = "X_" + dt
                task_data[k] = task_data[k].reshape(len(task_data[k]), -1)
        else:
            for dt in data_type:
                k = "X_" + dt
                if sequence_last_dim:
                    task_data[k] = task_data[k].reshape(len(task_data[k]), factor_num, -1)
                    if isinstance(task_data[k], torch.Tensor):
                        task_data[k] = task_data[k].permute(0, 2, 1)
                    else:
                        task_data[k] = task_data[k].transpose(0, 2, 1)
                else:
                    task_data[k] = task_data[k].reshape(len(task_data[k]), -1, factor_num)

        test_date = task_data["test_idx"].codes[0] - task_data["test_idx"].codes[0][0]
        task_data["meta_end"] = (test_date <= (test_date[-1] - H + 1)).sum()
    if skip:
        ''' Delete tasks with empty test data '''
        j = 0
        for idx, i in enumerate(skip):
            if i < j:
                continue
            j = i + 1
            k = idx + 1
            while j == skip[k]:
                k += 1
                j = skip[k]
            for key in ['X_train', 'y_train']:
                task_data_list[j][key] = task_data_list[i][key]
        task_data_list = [task_data_list[i] for i in range(len(task_data_list)) if i not in skip]
    return task_data_list


def override_state(groups, new_opt):
    saved_groups = new_opt.param_groups
    id_map = {old_id: p for old_id, p in zip(range(len(saved_groups[0]["params"])), groups[0]["params"])}
    state = defaultdict(dict)
    for k, v in new_opt.state[0].items():
        if k in id_map:
            param = id_map[k]
            for _k, _v in v.items():
                state[param][_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v
        else:
            state[k] = v
    return state


def has_rnn(module: nn.Module):
    for module in module.modules():
        if isinstance(module, nn.RNNBase):
            return True
    return False