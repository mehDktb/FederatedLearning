import argparse
import logging
import numpy as np
import os
from datasets import load_dataset
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ALBERT on a classification task")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to the training data file (.tsv or .csv with 'text' and 'label' columns)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to the test data file (.tsv or .csv with 'text' and 'label' columns)")
    parser.add_argument("--model_name_or_path", type=str, default="albert-base-v2",
                        help="Pretrained ALBERT model name or local path")
    parser.add_argument("--output_dir", type=str, default="./albert_fineted",
                        help="Where to save the final model and tokenizer")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--bsz", type=int, default=16, help="Batch size per device")
    parser.add_argument("--albert_dr", type=float, default=0.1, help="Dropout for ALBERT layers")
    parser.add_argument("--classifier_dr", type=float, default=0.1, help="Dropout for classification head")
    parser.add_argument("--ts", type=int, default=None,
                        help="Total training steps (overrides num_train_epochs if set)")
    parser.add_argument("--ws", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--msl", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        help="Learning rate scheduler type: constant, linear, cosine, etc.")
    return parser.parse_args()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # input data
    delim = '\t' if args.train_data.endswith('.tsv') else ','
    data_files = {"train": args.train_data, "test": args.test_data}
    raw_datasets = load_dataset('csv', data_files=data_files, delimiter=delim, split=['train', 'test'])
    train_dataset, eval_dataset = raw_datasets

    # ensure integer labels
    train_dataset = train_dataset.map(lambda ex: {"label": int(ex['label'])})
    eval_dataset  = eval_dataset.map(lambda ex: {"label": int(ex['label'])})

    # build config from model_name_or_path
    num_labels = len(set(train_dataset['label']))
    config = AlbertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        hidden_dropout_prob=args.albert_dr,
        classifier_dropout_prob=args.classifier_dr
    )

    # load model & tokenizer from model_name_or_path
    model = AlbertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    )
    tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)

    # print what we're using
    logging.info(f"Loaded model from {args.model_name_or_path}")
    logging.info(f"Will save outputs to {args.output_dir}")

    # preprocess
    def preprocess_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=args.msl
        )

    train_dataset = train_dataset.map(preprocess_fn, batched=True)
    eval_dataset  = eval_dataset.map(preprocess_fn, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # determine steps vs epochs
    if args.ts:
        max_steps = args.ts
        num_epochs = 1
    else:
        max_steps = -1
        num_epochs = 3

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        warmup_steps=args.ws,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        report_to='none'
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train + eval
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # save model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == '__main__':
    main()
