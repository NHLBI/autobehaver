"""Lightweight command-line inference execution options."""

def add_inference_runtime_arguments(parser):
    parser.add_argument('--memory_gb', type=float, default=None,
                        help='Total inference RAM budget in GiB; auto-detect allocation if omitted.')
    parser.add_argument('--noise_workers', type=int, default=0,
                        help='Noise preparation thread ceiling (0 = automatic from CPU/RAM).')
    parser.add_argument('--vram_fraction', type=float, default=.8,
                        help='VRAM calibration ceiling, not a filling target (default 0.8; maximum 0.9).')
    return parser
