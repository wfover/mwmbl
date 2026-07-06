"""Train the Super Search xgb source model and write its serving artifact.

Two data sources:

- ``--from-matrix PATH``  a dense offline ``RewardMatrix`` (``.npz`` +
  ``.json`` pair, e.g. ``devdata/ss_judge_matrix``) — used to build the
  repo-bundled warm-start artifact from the per-source home-query set.
- ``--days N``            logged ``SuperSearchImpression`` rows from the last
  N days — the same path the daily background retrain takes.

The artifact directory (``model.json`` + ``meta.json``) defaults to
``settings.SUPER_SEARCH_XGB_MODEL_DIR``; pass
``--out mwmbl/tinysearchengine/super_search_select/artifacts/xgb`` to refresh
the bundled warm-start model.
"""
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from mwmbl.tinysearchengine.super_search_select import xgb_model
from mwmbl.tinysearchengine.super_search_select.evaluation import RewardMatrix


class Command(BaseCommand):
    help = "Train the Super Search xgb source model (from a reward matrix or the impression log)"

    def add_arguments(self, parser):
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--from-matrix", metavar="PATH",
                            help="RewardMatrix path (without extension)")
        source.add_argument("--days", type=int,
                            help="train from SuperSearchImpression rows of the last N days")
        parser.add_argument("--min-rows", type=int,
                            default=settings.SUPER_SEARCH_XGB_MIN_TRAIN_ROWS)
        parser.add_argument("--out", default=settings.SUPER_SEARCH_XGB_MODEL_DIR,
                            help="artifact directory (default: SUPER_SEARCH_XGB_MODEL_DIR)")
        parser.add_argument("--reward-kind", default="judge",
                            help="provenance label stored in meta.json for --from-matrix")

    def handle(self, *args, **options):
        import numpy as np

        if options["from_matrix"]:
            matrix = RewardMatrix.load(options["from_matrix"])
            X, y, vocab = xgb_model.build_training_data_from_matrix(matrix)
            if len(y) < options["min_rows"]:
                raise CommandError(
                    f"only {len(y)} training pairs in the matrix, "
                    f"need {options['min_rows']} (--min-rows)")
            model = xgb_model.train(X, y)
            metrics = {"train_rmse": float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))}
            xgb_model.save_artifact(model, vocab, options["out"],
                                    reward_kind=options["reward_kind"],
                                    n_rows=len(y), metrics=metrics)
        else:
            metrics = xgb_model.train_and_save_from_impressions(
                window_days=options["days"],
                min_rows=options["min_rows"],
                out_dir=options["out"],
            )
            if metrics is None:
                raise CommandError("not enough impression pairs to train "
                                   "(see --min-rows)")
        self.stdout.write(self.style.SUCCESS(
            f"trained xgb source model -> {options['out']} ({metrics})"))
