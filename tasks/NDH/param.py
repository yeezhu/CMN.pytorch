import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_type', type=str, required=True,
                    help='planner_path, player_path, or trusted_path')
parser.add_argument('--history', type=str, required=True,
                    help='none, target, oracle_ans, nav_q_oracle_ans, or all')
parser.add_argument('--feedback', type=str, required=True,
                    help='teacher or sample')
parser.add_argument('--eval_type', type=str, required=True,
                    help='val or test')
parser.add_argument('--blind', action='store_true', required=False,
                    help='whether to replace the ResNet encodings with zero vectors at inference time')
parser.add_argument('--angle_feat_size', type=int, default=4)
parser.add_argument('--num_view', type=int, default=36)
parser.add_argument('--featdropout', type=float, default=0.3)
parser.add_argument('--ignoreid', type=int, default=-100)
parser.add_argument('--prefix', type=str, default="v1", required=True)
args = parser.parse_args()
