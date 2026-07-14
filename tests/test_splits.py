from dataclasses import dataclass

import pytest

from src.data.splits import grouped_train_validation_split


@dataclass
class Item:
    group: str
    value: int


def test_grouped_split_is_disjoint_and_reproducible():
    items = [Item(group, i) for i, group in enumerate("AABBCCDD")]
    train1, val1 = grouped_train_validation_split(
        items, validation_fraction=0.25, seed=7, group_key=lambda x: x.group
    )
    train2, val2 = grouped_train_validation_split(
        items, validation_fraction=0.25, seed=7, group_key=lambda x: x.group
    )
    assert [x.value for x in train1] == [x.value for x in train2]
    assert [x.value for x in val1] == [x.value for x in val2]
    assert {x.group for x in train1}.isdisjoint({x.group for x in val1})


def test_grouped_split_rejects_one_group():
    with pytest.raises(ValueError, match="at least two"):
        grouped_train_validation_split(
            [Item("A", 1)], validation_fraction=0.2, group_key=lambda x: x.group
        )
