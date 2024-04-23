import os
import torch
import logging
import traceback

from models.runner import DiffRunner
from utils.commons import parse_args_and_config

torch.set_printoptions(sci_mode=False)

if __name__ == "__main__":
    try:
        args, config = parse_args_and_config()
        logging.info("Writing log file to {}".format(args.log_path))
        logging.info("Exp instance id = {}".format(os.getpid()))

        runner = DiffRunner(args, config)
        assert os.path.exists(args.eval_path), f'Failed to locate the pretrained weight: {args.eval_path}'
        runner.create_diffusion_model(args.eval_path)
        runner.create_pose_model()
        runner.prepare_data(eval_only=True)
        runner.test(is_train=False, is_store=True)

    except Exception:
        logging.error(traceback.format_exc())
