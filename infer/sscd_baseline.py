import argparse
import logging
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
from vsc.baseline.score_normalization import score_normalize
from vsc.candidates import CandidateGeneration, MaxScoreAggregation
from vsc.index import VideoFeature
from vsc.metrics import (
    average_precision,
    AveragePrecision,
    CandidatePair,
    Dataset,
    Match,
)
from vsc.storage import load_features, store_features


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sscd_baseline.py")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--query_features",
    help="Path to query descriptors",
    type=str,
    default="./infer/outputs/swinv2_v115/test_query_sn.npz",
    required=False
)
parser.add_argument(
    "--ref_features",
    help="Path to reference descriptors",
    type=str,
    default="./infer/outputs/swinv2_v115/test_refs_sn.npz",  
    required=False
)
parser.add_argument(
    "--score_norm_features",
    help="Path to score normalization descriptors",
    type=str,
    default=None
)
parser.add_argument(
    "--output_path",
    help="The path to write match predictions.",
    type=str,
    default="./infer/output",  
    required=False
)
parser.add_argument(
    "--ground_truth",
    help="Path to the ground truth (labels) CSV file.",
    type=str,
    default="./data/meta/test/new_gt.csv" 
)
parser.add_argument(
    "--overwrite",
    help="Overwrite prediction files, if found.",
    action="store_true"
)

args = parser.parse_args()


def search(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    retrieve_per_query: float = 50.0,
    candidates_per_query: float = 10.0,
) -> List[CandidatePair]:
    aggregation = MaxScoreAggregation()
    logger.info("Searching")
    cg = CandidateGeneration(refs, aggregation)
    num_to_retrieve = int(retrieve_per_query * len(queries))
    print("len query:", len(queries))
    candidates = cg.query(queries, global_k=num_to_retrieve)
    num_candidates = int(candidates_per_query * len(queries))
    candidates = candidates[:num_candidates]
    logger.info("Got %d candidates", len(candidates))
    return candidates

def match(
    queries: List[VideoFeature],
    refs: List[VideoFeature],
    output_path: str,
    score_normalization: bool = False,
) -> Tuple[str, str]:
    # Search
    candidates = search(queries, refs)
    os.makedirs(output_path, exist_ok=True)
    candidate_file = os.path.join(output_path, "candidates.csv")
    CandidatePair.write_csv(candidates, candidate_file)
    return candidate_file


def create_pr_plot(ap: AveragePrecision, filename: str):
    ap.pr_curve.plot(linewidth=1)
    plt.savefig(filename)
    plt.show()


def main(args):
    if os.path.exists(args.output_path) and not args.overwrite:
        raise Exception(
            f"Output path already exists: {args.output_path}. Do you want to --overwrite?"
        )
    queries = load_features(args.query_features, Dataset.QUERIES)
    refs = load_features(args.ref_features, Dataset.REFS)
    score_normalization = False
    if args.score_norm_features:
        queries, refs = score_normalize(
            queries,
            refs,
            load_features(args.score_norm_features, Dataset.REFS),
            beta=1.2,
        )
        score_normalization = True
        os.makedirs(args.output_path, exist_ok=True)
        store_features(os.path.join(args.output_path, "sn_queries.npz"), queries)
        store_features(os.path.join(args.output_path, "sn_refs.npz"), refs)

    candidate_file = match(
        queries,
        refs,
        args.output_path,
        score_normalization=score_normalization,
    )

    if not args.ground_truth:
        return

    # Descriptor track uAP (approximate)
    gt_matches = Match.read_csv(args.ground_truth, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    print("gt_pairs:",gt_pairs)
    candidate_pairs = CandidatePair.read_csv(candidate_file)
    candidate_uap = average_precision(gt_pairs, candidate_pairs)
    logger.info(f"Candidate uAP: {candidate_uap.ap:.4f}")
    candidate_pr_file = os.path.join(args.output_path, "candidate_precision_recall.pdf")
    create_pr_plot(candidate_uap, candidate_pr_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
