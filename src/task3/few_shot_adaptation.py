"""Task 3 - Cross-room few-label adaptation, K in {1, 5, 10, 20}, A->B and B->A.

    python src/task3/few_shot_adaptation.py [--config configs/experiment.yaml]

Requires: build_dataset.py (windows) and task1/room_identification.py (router).

For every (target room, K, seed, feature extractor, classifier, routing mode):
  * shared SIT/STAND/WALK learned from the source room's labeled development captures,
  * target-exclusive activities from K target development captures per activity,
  * scored on ALL eligible validation windows of the target room.

Outputs (results/task3/):
  all_runs.csv               every run (long format)
  summary.csv                mean/std over support seeds: experiment,target_room,K,metric,value,std
  label_efficiency.png       macro-F1 vs K, overall / target-exclusive / shared (headline system)
  encoder_comparison.png     SSL vs random vs handcrafted features
  routing_contribution.png   predicted vs oracle vs no routing
  confusion_K5_target_{A,B}.png + per_class_K5_target_{A,B}.csv
  predictions_K5_seed0.csv   window_id, room, label, confidence (+ true label)
and models/deploy.joblib used by run_episode.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.adaptation import (  # noqa: E402
    FeatureBank, activity_metrics, fit_target_model, predict, select_support, source_rows,
)
from pipeline.config import ROOM_LABELS, load_config, other_room  # noqa: E402
from pipeline.data import load_windows  # noqa: E402
from pipeline.features import FeatureExtractor  # noqa: E402
from pipeline.ingest import load_normalizers  # noqa: E402


def direction(target):
    return f"{other_room(target)}->{target}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    t3 = cfg["task3"]
    shared = t3["shared_activities"]

    data = load_windows(cfg)
    meta = data.meta
    normalizers = load_normalizers(cfg)
    router = joblib.load(cfg.path("models") / "room_router.joblib")["router"]
    val_index = np.flatnonzero((meta.split == "validation").to_numpy())
    routed_all = np.empty(len(meta), dtype=object)
    routed_all[val_index], _ = router.predict(data.windows[val_index])
    print("router validation accuracy:",
          float(np.mean(routed_all[val_index] == meta.room.to_numpy()[val_index])))

    out = cfg.path("results") / "task3"
    out.mkdir(parents=True, exist_ok=True)

    banks = {}
    for kind in t3["feature_extractors"]:
        started = time.time()
        banks[kind] = FeatureBank(data.windows, meta, normalizers, FeatureExtractor(kind, cfg),
                                  alignment=bool(t3["feature_alignment"]))
        print(f"features[{kind}] computed in {time.time() - started:.1f}s")

    rows, prediction_rows, confusions = [], [], {}
    default_ext, default_clf = t3["default_extractor"], t3["default_classifier"]
    for kind, bank in banks.items():
        for classifier in t3["classifiers"]:
            for k in t3["k_values"]:
                for seed in t3["seeds"]:
                    support = {r: select_support(meta, r, t3["target_exclusive"][r], k, seed) for r in "AB"}
                    for mode in t3["routing_modes"]:
                        fit_mode = "none" if mode == "none" else "routed"
                        models = {r: fit_target_model(bank, r, support[r], shared, classifier, fit_mode)
                                  for r in "AB"}
                        for target in "AB":
                            index = val_index[(meta.room.to_numpy()[val_index] == target)]
                            if mode == "predicted":
                                routed = routed_all[index]
                            else:  # oracle: true room; none: evaluated target, pooled domain
                                routed = np.array([target] * len(index), dtype=object)
                            labels, conf = predict(bank, models, index, routed,
                                                   "none" if mode == "none" else "routed")
                            truth = meta.activity.to_numpy()[index]
                            metrics = activity_metrics(truth, labels, ROOM_LABELS[target],
                                                       t3["target_exclusive"][target], shared)
                            for name, value in metrics.items():
                                rows.append({"experiment": "task3_few_shot", "direction": direction(target),
                                             "target_room": target, "K": k, "seed": seed,
                                             "extractor": kind, "classifier": classifier,
                                             "routing": mode, "metric": name, "value": value,
                                             "n_validation": len(index)})
                            headline = kind == default_ext and classifier == default_clf and mode == "predicted"
                            if headline and k == t3["confusion_k"]:
                                confusions.setdefault(target, []).append((truth, labels))
                                if seed == t3["seeds"][0]:
                                    prediction_rows.append(pd.DataFrame({
                                        "window_id": meta.window_id.to_numpy()[index], "room": routed,
                                        "label": labels, "confidence": conf, "true_room": target,
                                        "true_label": truth}))
                print(f"  {kind:11s} {classifier:9s} K={k:2d} done")

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "all_runs.csv", index=False)
    keys = ["experiment", "direction", "target_room", "K", "extractor", "classifier", "routing", "metric"]
    summary = runs.groupby(keys).value.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "value", "count": "n_seeds"})
    summary.to_csv(out / "summary.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows).to_csv(out / f"predictions_K{t3['confusion_k']}_seed0.csv", index=False)

    headline = summary[(summary.extractor == default_ext) & (summary.classifier == default_clf)
                       & (summary.routing == "predicted")]
    table = headline.pivot_table(index=["direction", "K"], columns="metric", values="value")
    print("\nHEADLINE (extractor=%s, classifier=%s, predicted routing) - validation:" % (default_ext, default_clf))
    print(table[["macro_f1", "target_exclusive_macro_f1", "shared_macro_f1", "balanced_accuracy"]].round(4))
    table.round(4).to_csv(out / "headline_table.csv")

    plot_label_efficiency(headline, t3, out)
    plot_comparison(summary, t3, out, "extractor", t3["feature_extractors"],
                    dict(classifier=default_clf, routing="predicted"), "encoder_comparison.png",
                    "Feature extractor comparison (macro-F1, predicted routing)")
    plot_comparison(summary, t3, out, "routing", t3["routing_modes"],
                    dict(classifier=default_clf, extractor=default_ext), "routing_contribution.png",
                    "Routing contribution (macro-F1)")
    plot_comparison(summary, t3, out, "classifier", t3["classifiers"],
                    dict(extractor=default_ext, routing="predicted"), "classifier_comparison.png",
                    "Classifier comparison (macro-F1, predicted routing)")
    for target, pairs in confusions.items():
        plot_confusion(target, pairs, t3, out)

    save_deploy_bundle(cfg, banks[default_ext], meta, router, normalizers, t3)
    print("\nSaved results to", out)


def plot_label_efficiency(headline, t3, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, target in zip(axes, "AB"):
        part = headline[headline.target_room == target]
        for metric, label in (("macro_f1", "all target activities"),
                              ("target_exclusive_macro_f1", "target-labeled (exclusive)"),
                              ("shared_macro_f1", "transferred shared (SIT/STAND/WALK)")):
            series = part[part.metric == metric].sort_values("K")
            ax.errorbar(series.K, series.value, yerr=series["std"].fillna(0), marker="o", capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_xticks(t3["k_values"])
        ax.set_xticklabels(t3["k_values"])
        ax.set_title(f"Target room {target} ({direction(target)})")
        ax.set_xlabel("K labeled target captures per exclusive activity")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Label efficiency ({t3['default_extractor']} features, {t3['default_classifier']}, "
                 "predicted routing; mean ± std over support draws)")
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=200)
    plt.close(fig)


def plot_comparison(summary, t3, out, column, values, fixed, filename, title):
    part = summary[summary.metric == "macro_f1"]
    for key, value in fixed.items():
        part = part[part[key] == value]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, target in zip(axes, "AB"):
        for value in values:
            series = part[(part.target_room == target) & (part[column] == value)].sort_values("K")
            ax.errorbar(series.K, series.value, yerr=series["std"].fillna(0), marker="o", capsize=3, label=value)
        ax.set_xscale("log")
        ax.set_xticks(t3["k_values"])
        ax.set_xticklabels(t3["k_values"])
        ax.set_title(f"Target room {target} ({direction(target)})")
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(title=column, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def plot_confusion(target, pairs, t3, out):
    labels = ROOM_LABELS[target]
    truth = np.concatenate([p[0] for p in pairs])
    pred = np.concatenate([p[1] for p in pairs])
    shown = labels + sorted(set(pred) - set(labels))   # mis-routed predictions appear as extra columns
    cm = confusion_matrix(truth, pred, labels=shown)[: len(labels)]
    p, r, f, s = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    pd.DataFrame({"activity": labels, "precision": p, "recall": r, "f1": f, "support": s}).to_csv(
        out / f"per_class_K{t3['confusion_k']}_target_{target}.csv", index=False)
    fig, ax = plt.subplots(figsize=(1.1 * len(shown) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(shown, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l}  (P={p_:.2f} R={r_:.2f} F1={f_:.2f})" for l, p_, r_, f_ in zip(labels, p, r, f)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Target room {target}, K={t3['confusion_k']} (summed over {len(pairs)} support draws)")
    fig.tight_layout()
    fig.savefig(out / f"confusion_K{t3['confusion_k']}_target_{target}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_deploy_bundle(cfg, bank, meta, router, normalizers, t3):
    """Models for run_episode.py: K = deploy_k, first seed, headline configuration."""
    k, seed = int(t3["deploy_k"]), int(t3["seeds"][0])
    models, training = {}, {}
    for target in "AB":
        support = select_support(meta, target, t3["target_exclusive"][target], k, seed)
        models[target] = fit_target_model(bank, target, support, t3["shared_activities"],
                                          t3["default_classifier"], "routed")
        src = source_rows(meta, target, t3["shared_activities"])
        training[target] = {"source_x": bank.get(src, other_room(target)),
                            "source_y": meta.activity.to_numpy()[src],
                            "support_captures": meta.capture_id.to_numpy()[support].tolist()}
    bundle = {
        "preprocessing_hash": cfg.preprocessing_hash,
        "extractor": t3["default_extractor"],
        "classifier": t3["default_classifier"],
        "feature_alignment": bool(t3["feature_alignment"]),
        "feature_stats": {d: bank.stats[d] for d in ("A", "B")},
        "normalizers": {d: (normalizers[d].mean, normalizers[d].std) for d in ("A", "B", "pooled")},
        "router": router,
        "models": models,
        "training": training,
        "deploy_k": k,
        "shared": t3["shared_activities"],
        "target_exclusive": t3["target_exclusive"],
    }
    path = cfg.path("models") / "deploy.joblib"
    joblib.dump(bundle, path)
    print("Deployment bundle:", path)


if __name__ == "__main__":
    main()
