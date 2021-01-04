import os, csv
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NsmcProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):

        fr = open(input_file, 'r', encoding='utf-8', errors="ignore")
        rdr = csv.reader(fr, delimiter='\t')

        data = [x for x in rdr]
        fr.close()
        return data

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        examples_id = []
        for (i, line) in enumerate(lines[1:]):
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = line[3]
            sentence_order = int(line[4])
            if set_type == 'test':
                label = 0
                examples_id.append(line[5])
            else:
                label = int(line[5])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples, examples_id

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "nsmc": NsmcProcessor,
}


def convert_examples_to_features(examples, max_title_len, max_sentence_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        # '''token a feature'''
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens_a) > max_title_len - special_tokens_count:
            tokens_a = tokens_a[:(max_title_len - special_tokens_count)]

        # Add [SEP] token
        tokens_a += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens_a)

        # Add [CLS] token
        tokens_a = [cls_token] + tokens_a
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask_a = [1 if mask_padding_with_zero else 0] * len(input_ids_a)

        # Zero-pad up to the sequence length.
        padding_length = max_title_len - len(input_ids_a)
        input_ids = input_ids_a + ([pad_token_id] * padding_length)
        attention_mask = attention_mask_a + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([sequence_a_segment_id] * padding_length)

        # '''token b feature'''
        # Account for [SEP]
        special_tokens_count = 1
        if len(tokens_b) > max_sentence_len - special_tokens_count:
            tokens_b = tokens_b[:(max_sentence_len - special_tokens_count)]

        # Add [SEP] token
        tokens_b += [sep_token]
        token_type_ids += [sequence_b_segment_id] * len(tokens_b)

        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        attention_mask_b = [1 if mask_padding_with_zero else 0] * len(input_ids_b)

        # Zero-pad up to the sequence length.
        padding_length = max_sentence_len - len(input_ids_b)
        input_ids += input_ids_b + ([pad_token_id] * padding_length)
        attention_mask += attention_mask_b + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([sequence_b_segment_id] * padding_length)


        assert len(input_ids) == max_title_len + max_sentence_len, "Error with input length {} vs {}".format(
            len(input_ids), max_title_len)
        assert len(
            attention_mask) == max_title_len + max_sentence_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_title_len + max_sentence_len)
        assert len(token_type_ids) == max_title_len + max_sentence_len, "Error with token type length {} vs {}".format(
            len(token_type_ids),
            max_title_len + max_sentence_len)

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_a + tokens_b]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(),
        args.max_title_len + args.max_sentence_len, mode)

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file) and not (mode == 'test'):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples, _ = processor.get_examples("train")
        elif mode == "dev":
            examples, _ = processor.get_examples("dev")
        elif mode == "test":
            examples, examples_id = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_title_len, args.max_sentence_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    if mode == 'test':
        return dataset, examples_id

    return dataset
