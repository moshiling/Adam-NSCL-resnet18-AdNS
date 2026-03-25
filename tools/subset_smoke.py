import argparse
import os
import sys
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import main as main_mod


def run_subset_smoke():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--projection_mode", required=True)
    parser.add_argument("--task_limit", type=int, default=2)
    parser.add_argument("--subset_samples", type=int, default=256)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    cli = parser.parse_args()

    passthrough = list(cli.extra_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    args = main_mod.get_args(
        ["--config", cli.config, "--experiment_name", cli.experiment_name] + passthrough
    )

    run_dir = main_mod.ensure_dir(os.path.join(PROJECT_ROOT, args.output_root, args.experiment_name))
    main_mod.ensure_dir(os.path.join(run_dir, "checkpoints"))
    main_mod.save_json(os.path.join(run_dir, "config_resolved.json"), vars(args))

    if torch.cuda.is_available() and args.gpuid[0] >= 0:
        torch.cuda.set_device(args.gpuid[0])
        use_cuda = True
    else:
        args.gpuid = [-1]
        use_cuda = False
    main_mod.set_random_seed(args.seed, use_cuda)

    train_dataset_splits, val_dataset_splits, task_output_space = main_mod.prepare_datasets(args)
    task_names = sorted(list(task_output_space.keys()), key=int)[: cli.task_limit]
    task_output_space = OrderedDict((name, task_output_space[name]) for name in task_names)
    agent = main_mod.create_agent(args, task_output_space, run_dir)

    acc_table = OrderedDict()
    task_metrics_history = []
    plasticity_history = []
    start_task_idx = 0
    if cli.resume_checkpoint:
        checkpoint = torch.load(cli.resume_checkpoint, map_location="cpu")
        agent.load_serialized_state(checkpoint["agent_state"])
        loaded_acc_table = checkpoint.get("acc_table", {})
        for outer_key in task_names:
            if outer_key in loaded_acc_table:
                acc_table[outer_key] = OrderedDict(loaded_acc_table[outer_key])
        task_metrics_history = checkpoint.get("task_metrics_history", [])
        plasticity_history = checkpoint.get("plasticity_history", [])
        start_task_idx = checkpoint.get("next_task_idx", 0)

    for i, train_name in enumerate(task_names):
        if i < start_task_idx:
            continue
        train_subset = Subset(
            train_dataset_splits[train_name],
            list(range(min(cli.subset_samples, len(train_dataset_splits[train_name])))),
        )
        val_subset = Subset(
            val_dataset_splits[train_name],
            list(range(min(cli.subset_samples, len(val_dataset_splits[train_name])))),
        )
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        train_metrics = agent.train_task(train_loader, val_loader, task_name=train_name)
        torch.cuda.empty_cache()
        acc_table[train_name] = OrderedDict()
        for j in range(i + 1):
            val_name = task_names[j]
            eval_subset = Subset(
                val_dataset_splits[val_name],
                list(range(min(cli.subset_samples, len(val_dataset_splits[val_name])))),
            )
            eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            acc_table[val_name][train_name] = agent.validation(eval_loader)

        avg_acc_history, bwt_history = main_mod.compute_histories(acc_table, task_names[: i + 1])
        task_summary = {
            "task": train_name,
            "task_index": i + 1,
            "avg_acc": avg_acc_history[-1],
            "bwt": bwt_history[-1],
            "acc_by_eval_task": {name: acc_table[name][train_name] for name in task_names[: i + 1]},
            "train_metrics": train_metrics,
            "projection_build_stats": getattr(agent.model_optimizer, "latest_build_stats", {}),
            "subset_smoke_samples": cli.subset_samples,
            "limited_tasks": cli.task_limit,
        }
        task_metrics_history.append(task_summary)
        plasticity_payload = {
            "task": train_name,
            "task_index": i + 1,
            "projection_mode": cli.projection_mode,
            "alpha_t": train_metrics.get("alpha_t", 1.0),
            "rho_t": train_metrics.get("rho_t", 1.0),
            "optimizer_task_stats": train_metrics.get("optimizer_task_stats", {}),
            "projection_build_stats": getattr(agent.model_optimizer, "latest_build_stats", {}),
            "subset_smoke_samples": cli.subset_samples,
            "limited_tasks": cli.task_limit,
        }
        plasticity_history.append(plasticity_payload)
        main_mod.append_jsonl(os.path.join(run_dir, "task_metrics.jsonl"), task_summary)
        main_mod.append_jsonl(os.path.join(run_dir, "plasticity_stats.jsonl"), plasticity_payload)
        teacher_warmup = train_metrics.get("teacher_warmup", [])
        if teacher_warmup:
            main_mod.save_json(
                os.path.join(run_dir, "teacher_warmup_metrics_task_{:02d}.json".format(i + 1)),
                {
                    "task": train_name,
                    "task_index": i + 1,
                    "teacher_warmup_metrics": teacher_warmup,
                    "subset_smoke_samples": cli.subset_samples,
                },
            )
        for epoch_payload in train_metrics.get("epoch_metrics", []):
            main_mod.append_jsonl(
                os.path.join(run_dir, "train_loss_breakdown.jsonl"),
                {
                    "task": train_name,
                    "task_index": i + 1,
                    "epoch": epoch_payload.get("epoch"),
                    "acc": epoch_payload.get("acc"),
                    "ce_loss": epoch_payload.get("ce_loss"),
                    "distill_loss": epoch_payload.get("distill_loss"),
                    "total_loss": epoch_payload.get("total_loss", epoch_payload.get("loss")),
                    "loss": epoch_payload.get("loss"),
                    "val_acc": epoch_payload.get("val_acc"),
                    "subset_smoke_samples": cli.subset_samples,
                },
            )
        main_mod.save_checkpoint(
            os.path.join(run_dir, "checkpoints", "last.pt"),
            agent,
            acc_table,
            task_names,
            i + 1,
            task_metrics_history,
            plasticity_history,
            args,
        )
        main_mod.save_checkpoint(
            os.path.join(run_dir, "checkpoints", "task_{:02d}.pt".format(i + 1)),
            agent,
            acc_table,
            task_names,
            i + 1,
            task_metrics_history,
            plasticity_history,
            args,
        )

    avg_acc_history, bwt_history = main_mod.compute_histories(acc_table, task_names)
    summary = {
        "run_name": args.experiment_name,
        "run_dir": run_dir,
        "projection_mode": cli.projection_mode,
        "task_order": task_names,
        "acc_table": acc_table,
        "avg_acc_history": avg_acc_history,
        "bwt_history": bwt_history,
        "final_avg_acc": avg_acc_history[-1] if avg_acc_history else 0.0,
        "final_bwt": bwt_history[-1] if bwt_history else 0.0,
        "seed": args.seed,
        "gpuid": args.gpuid,
        "limited_tasks": cli.task_limit,
        "subset_smoke_samples": cli.subset_samples,
        "resume_checkpoint": cli.resume_checkpoint,
    }
    main_mod.save_json(os.path.join(run_dir, "summary.json"), summary)
    print(summary["final_avg_acc"])
    print(summary["final_bwt"])


if __name__ == "__main__":
    run_subset_smoke()
