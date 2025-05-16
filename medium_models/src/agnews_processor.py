"""Processor for the AGNews dataset."""

import os
import logging
import pandas as pd
from transformers import DataProcessor, InputExample

logger = logging.getLogger(__name__)

class AGNewsProcessor(DataProcessor):
    """Processor for the AGNews dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]  # AGNews has 4 classes

    def _create_examples(self, df, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, row) in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = row['text']
            label = str(row['label'])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples 