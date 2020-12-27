import argparse, os

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, write_csvFile
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, dataset_id = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train(mode="train")
    if args.do_dev:
        trainer.train(mode="dev")

    if args.do_eval:
        trainer.load_model()
        results = trainer.evaluate("test")
        print("dataset_id : ", dataset_id)
        print("results : ", results)
        results = [[data_id, result] for (data_id, result) in zip(dataset_id, results)]
        print(results)
        write_csvFile(os.path.join(args.data_dir, "result.csv"), results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="nsmc", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="news_train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="news_test_temp.tsv", type=str, help="Test file")
    parser.add_argument("--test_file", default="news_test.tsv", type=str, help="Test file")

    parser.add_argument("--model_type", default="hanbert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="HanBert-54kN-torch", type=str,
                        help="Path to pre-trained model or shortcut name")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_title_len", default=50, type=int,
                        help="The maximum title input sequence length after tokenization.")
    parser.add_argument("--max_sentence_len", default=100, type=int,
                        help="The maximum sentence of contents input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_dev", action="store_true", help="Whether to run for dev.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    main(args)
