import argparse
import os
import torch
from .utils import mults


class TokensToSeqs(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        assert "tokens" in dest, f"Argument {dest} must contain 'tokens'"
        if nargs is not None:
            raise ValueError("nargs not allowed for TokensToStep")
        super(TokensToSeqs, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):

        seq_len = namespace.max_seq_len
        token_str = values.lower()
        num, mult = token_str[:-1], token_str[-1]
        assert (
            mult in mults
        ), f"{self.dest} must be specified in tokens with a suffix(K,M,B)"
        tokens = int(num) * mults[mult]
        assert (
            tokens % seq_len == 0
        ), f"{self.dest} token count must be divisible by seq_len"
        setattr(namespace, self.dest, tokens)
        setattr(namespace, self.dest.replace("tokens", "seqs"), int(tokens // seq_len))


def get_training_arguments():
    parser = argparse.ArgumentParser(description="Transformer Training Arguments")

    # Model hyperparameters
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="default max sequence length"
    )
    parser.add_argument(
        "--n-layers", type=int, default=20, help="Number of transformer layers"
    )
    parser.add_argument(
        "--d-model", type=int, default=2048, help="Dimensionality of the model"
    )
    parser.add_argument(
        "--n-heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-q-heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-kv-heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=None,
        help="Attention head dim. If None, d_model // n_heads",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=8192,
        help="Dimensionality of the feed-forward layer",
    )
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--glu", action="store_true", help="Enable GLU")
    # parser.add_argument('--rope', action='store_true', help='Enable RoPE')

    # Training settings
    parser.add_argument("--micro-batch-tokens", action=TokensToSeqs, help="Batch size")
    parser.add_argument("--global-batch-tokens", action=TokensToSeqs, help="Batch size")
    parser.add_argument(
        "--num-tokens", action=TokensToSeqs, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.01, help="warmup ratio")
    # Optimization settings
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Relative weight decay"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument(
        "--beta2", type=float, default=0.95, help="Adam optimizer beta2"
    )

    # Logging/Saving arguments
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm")
    parser.add_argument(
        "--tqdm-flops",
        action="store_true",
        help="Have tqdm log flops instead of tokens",
    )
    parser.add_argument(
        "--log-interval-tokens", action=TokensToSeqs, help="Logging interval"
    )
    parser.add_argument(
        "--save-interval-tokens", action=TokensToSeqs, help="Saving interval"
    )
    parser.add_argument(
        "--output-folder", default=None, type=str, required=True, help="output folder"
    )
    parser.add_argument(
        "--resume", action="store_true", help="load existing checkpoint"
    )
    parser.add_argument(
        "--restart", action="store_true", help="load existing checkpoint"
    )

    # Torch distributed arguments
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/dataset/bluepile/dolma/processed/data",
        help="data location",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    # assert (
    #     args.save_interval_seqs % args.global_batch_seqs == 0
    # ), "save interval must be divisible by global batch size"
    # assert (
    #     args.log_interval_seqs % args.global_batch_seqs == 0
    # ), "log interval must be divisible by global batch size"

    # # Attention Argument handling
    # if args.d_head is None:
    #     args.d_head = args.d_model // args.n_heads
    # if args.n_q_heads is None:
    #     args.n_q_heads = args.n_heads
    # if args.n_kv_heads is None:
    #     args.n_kv_heads = args.n_heads

    if torch.distributed.get_rank() == 0:
        if not os.path.isdir(args.output_folder):
            print("making directory", args.output_folder)
            os.makedirs(args.output_folder)
        # else:
        #     assert (
        #         args.resume or args.restart
        #     ), "continuing an existing folder without specifying resume or restart. Are you sure?"

    return args


if __name__ == "__main__":
    args = get_training_arguments()
    print(args)
