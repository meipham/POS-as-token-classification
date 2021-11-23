import logging
import sys
import argparse

import transformers
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from artifact import *
from pos_data_loader import load_pos_dataset
from tmt_pos_tagger import TMTPosTaggingModel
from tmt_pos_trainer import TMTPosTrainer

import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        default="",
        help="Training data file.",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="",
        help="Validating data file.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="",
        help="Test data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Model output directory.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Pretrained model for evaluation or predict"
    )
    parser.add_argument("--do_train", type=bool, help="Do training or not."),
    parser.add_argument("--do_eval", type=bool, help="Do validation or not."),
    parser.add_argument("--do_predict", type=bool, help="Do prediction or not."),

    parser.add_argument("--max_sequence_length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=282)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=64)
    parser.add_argument("--logging_first_step", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default='steps')
    parser.add_argument("--save_steps", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__== "__main__":
    args = parse_args()

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Set the verbosity to info of the Transformers logger (on main process only):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    #----------------# Load datasets #----------------#

    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file

    data_files = {
            "train": train_file, 
            "dev": dev_file, 
            "test": test_file
            }

    max_sequence_length = args.max_sequence_length
    padding = "max_length"

    dataset = load_pos_dataset(data_files, padding=padding, max_length=max_sequence_length)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Examples: {train_dataset[index]}.")

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validate set size: {len(dev_dataset)}")


    ######## Initialize model ########
    #----------------# Training #----------------#

    model = None

    if args.do_train is True: 
        last_checkpoint = None
        last_checkpoint = get_last_checkpoint(args.output_dir)
        model = TMTPosTaggingModel(last_checkpoint if last_checkpoint is not None else PRETRAINED_BERT_DIR, num_labels=len(TAGSET))

    #----------------# Evaluate #----------------#

    if args.do_eval is True or args.do_predict is True:
        model = TMTPosTaggingModel(args.model_dir)
  

    #----------------# Inference #----------------#

    trainer = TMTPosTrainer(
        model = model,
        args = TrainingArguments(
            output_dir = args.output_dir,
            overwrite_output_dir = True,
            do_train = args.do_train,
            do_eval = args.do_eval,
            do_predict = args.do_predict,
            seed = args.seed,
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs,
            per_device_train_batch_size = args.per_device_train_batch_size,
            per_device_eval_batch_size = args.per_device_eval_batch_size,
            logging_steps = args.logging_steps,
            logging_first_step = args.logging_first_step,
            evaluation_strategy = args.evaluation_strategy,
            save_steps = args.save_steps,
            logging_dir='./logs'
        ),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    ### MAIN TASK ###
    if args.do_train is True: 
        trainer.train(resume_from_checkpoint=last_checkpoint)
    
    if args.do_eval is True: 
        evaluation = trainer.evaluate (test_dataset)
        print(evaluation)
    
    if args.do_predict is True:
        prediction = trainer.predict(test_dataset)
        print(prediction)