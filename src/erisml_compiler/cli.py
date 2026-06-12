"""ErisML Compiler CLI.

Subcommands:
    compile <input>  --out <ir.json>  [--tier ...] [--extractor ...] [--stream]
    validate <ir.json>
    rlef <ir.json>  --out <rlef.json>
    report <ir.json>  --out <report.html>
    bundle <ir.json> --out <bundle-dir/>
    version
"""
from __future__ import annotations

from pathlib import Path

import click

from erisml_compiler import __schema_version__, __version__
from erisml_compiler.audit.artifact import bundle_artifact
from erisml_compiler.export.json_export import export_json, load_json
from erisml_compiler.export.rlef import export_rlef
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.streaming.captioner import TerminalCaptioner
from erisml_compiler.streaming.streamer import MoralStreamer
from erisml_compiler.tiers import CompilerTier
from erisml_compiler.viz.html_report import render_html_report


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """ErisML Compiler: structure-preserving compiler from natural language to ErisML."""


# ----------------------------------------------------------------- compile

@cli.command("compile")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output JSON IR path. Default: out/<stem>.json")
@click.option("--tier", type=click.Choice([t.value for t in CompilerTier]), default=None,
              help="Compiler tier. Default: auto-detect from extension.")
@click.option("--extractor", type=click.Choice(["mock", "rule", "llm"]), default="rule",
              help="Extractor backend (ignored for tier=geometric).")
@click.option("--critic", type=click.Choice(["mock", "rule", "llm"]), default=None,
              help="Critic extractor for cross-extractor consensus. Default: none.")
@click.option("--em-profile", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="EM-DAG YAML profile. Default: bundled default profile.")
@click.option("--canonicalizer", type=click.Choice(["auto", "registry", "labse"]), default="auto",
              help="Canonicalizer backend. 'auto' picks LaBSE if available, registry otherwise.")
@click.option("--stream", is_flag=True, help="Stream real-time captions to stdout.")
def cmd_compile(
    input_file: Path,
    out_path: Path | None,
    tier: str | None,
    extractor: str,
    critic: str | None,
    em_profile: Path | None,
    canonicalizer: str,
    stream: bool,
) -> None:
    """Compile a document to an IR JSON file."""
    if tier is None:
        tier_enum = CompilerTier.auto_detect(input_file)
    else:
        tier_enum = CompilerTier(tier)
    if out_path is None:
        out_path = Path("out") / f"{input_file.stem}.json"

    # Resolve canonicalizer.
    if canonicalizer == "registry":
        from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
        canon = RegistryCanonicalizer()
    elif canonicalizer == "labse":
        from erisml_compiler.canonicalizer.labse import LaBSECanonicalizer
        canon = LaBSECanonicalizer()
    else:
        from erisml_compiler.canonicalizer.base import auto_canonicalizer
        canon = auto_canonicalizer()

    click.echo(f"[#] Tier: {tier_enum.description}")
    click.echo(f"[#] Input: {input_file}")
    click.echo(f"[#] Extractor: {extractor}{f'  (critic: {critic})' if critic else ''}")
    click.echo(f"[#] Canonicalizer: {canon.name}")
    click.echo(f"[#] Output: {out_path}")

    ir = compile_document(
        input_file,
        CompileOptions(
            tier=tier_enum, extractor=extractor, critic=critic,
            em_profile=em_profile, canonicalizer=canon,
        ),
    )
    export_json(ir, out_path)
    click.echo(f"[+] Wrote IR: {out_path}")
    click.echo(f"[+] Canonical form: {ir.canonical_form}")
    if ir.deme_verdict:
        click.echo(f"[+] Verdict: {ir.deme_verdict.verdict}  (confidence {ir.deme_verdict.confidence:.2f})")
    if ir.audit:
        click.echo(f"[+] IR hash: {ir.audit.ir_hash}")

    if stream:
        click.echo("\n---- Stream ----")
        TerminalCaptioner().render(MoralStreamer(ir))


# ----------------------------------------------------------------- validate

@cli.command("validate")
@click.argument("ir_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def cmd_validate(ir_file: Path) -> None:
    """Validate an IR JSON file against the current Pydantic schemas."""
    try:
        ir = load_json(ir_file)
    except Exception as exc:
        click.echo(f"[-] Validation FAILED: {exc}", err=True)
        raise SystemExit(1)
    click.echo(f"[+] Valid IR ({ir.schema_version}). {len(ir.stakeholders)} stakeholders, "
               f"{len(ir.commitments)} commitments, {len(ir.ethical_facts)} ethical facts.")


# ----------------------------------------------------------------- rlef

@cli.command("rlef")
@click.argument("ir_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output RLEF JSON path.")
@click.option("--corrections", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Optional human-corrections JSON to bundle into the record.")
def cmd_rlef(ir_file: Path, out_path: Path, corrections: Path | None) -> None:
    """Export an IR as an RLEF training record."""
    import json
    ir = load_json(ir_file)
    human = json.loads(corrections.read_text(encoding="utf-8")) if corrections else None
    export_rlef(ir, out_path, human_corrections=human)
    click.echo(f"[+] Wrote RLEF record: {out_path}")


# ----------------------------------------------------------------- diff

@cli.command("diff")
@click.argument("old_ir", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("new_ir", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Optional: write machine-readable diff JSON to this path.")
def cmd_diff(old_ir: Path, new_ir: Path, json_out: Path | None) -> None:
    """Diff two IR JSON files. Human-readable on stdout; optional JSON output."""
    import json
    from erisml_compiler.correction.diff import diff_irs

    old = load_json(old_ir)
    new = load_json(new_ir)
    diff = diff_irs(old, new)
    if diff.is_empty:
        click.echo("[=] No structural differences (audit and timestamps excluded).")
    else:
        click.echo(f"[~] Diff: {old_ir} -> {new_ir}")
        click.echo(f"    scalar changes: {len(diff.scalar_changes)}")
        click.echo(f"    entity diffs:   {len(diff.entity_diffs)}")
        for line in diff.summary_lines():
            click.echo(line)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(diff.as_dict(), indent=2), encoding="utf-8")
        click.echo(f"[+] Diff JSON: {json_out}")


# ----------------------------------------------------------------- correct

@cli.command("correct")
@click.argument("ir_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("corrections_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output corrected-IR JSON path.")
@click.option("--corrector-id", type=str, default=None,
              help="Override the corrector_id field in the corrections file.")
@click.option("--reevaluate", is_flag=True,
              help="Re-run EM-DAG + DEME after corrections to re-evaluate the IR.")
def cmd_correct(
    ir_file: Path,
    corrections_file: Path,
    out_path: Path,
    corrector_id: str | None,
    reevaluate: bool,
) -> None:
    """Apply a corrections JSON to an IR. Records the correction in the audit trail."""
    from erisml_compiler.correction.corrector import Corrector
    ir = load_json(ir_file)
    new_ir, record, summaries = Corrector(ir).apply(corrections_file, corrector_id=corrector_id)
    for line in summaries:
        click.echo(line)

    if reevaluate:
        from erisml_compiler.em_dag import load_profile
        from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
        from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
        from erisml_compiler.evaluation.tensor_builder import build_timeline
        bundled = Path(__file__).parent / "em_dag" / "profiles" / "default.yaml"
        dag = load_profile(bundled)
        new_ir.em_outputs = dag.evaluate(new_ir)
        new_ir.timeline = build_timeline(new_ir, dag)
        final_vector = build_moral_vector_from_em_outputs(new_ir.em_outputs, dag)
        new_ir.moral_vectors = [final_vector]
        new_ir.deme_verdict = DEMEBridge().evaluate(new_ir, final_vector)
        click.echo(f"[+] Re-evaluated. New verdict: {new_ir.deme_verdict.verdict}")

    export_json(new_ir, out_path)
    click.echo(f"[+] Wrote corrected IR: {out_path}")
    click.echo(f"[+] {record.n_patches_applied} patches applied, {record.n_patches_failed} failed.")
    click.echo(f"[+] Pre  hash: {record.pre_correction_ir_hash}")
    click.echo(f"[+] Post hash: {record.post_correction_ir_hash}")


# ----------------------------------------------------------------- report

@cli.command("report")
@click.argument("ir_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output HTML report path.")
@click.option("--timeline-png", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Optional MoralVector timeline PNG to embed (will be generated if matplotlib available).")
def cmd_report(ir_file: Path, out_path: Path, timeline_png: Path | None) -> None:
    """Render a self-contained HTML report from an IR."""
    ir = load_json(ir_file)
    if timeline_png is None:
        try:
            from erisml_compiler.viz.timeline_plot import save_timeline_plot
            timeline_png = out_path.with_suffix(".png")
            save_timeline_plot(ir, timeline_png)
        except ImportError:
            timeline_png = None
    html = render_html_report(ir, timeline_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"[+] Wrote HTML report: {out_path}")
    if timeline_png:
        click.echo(f"[+] Timeline PNG: {timeline_png}")


# ----------------------------------------------------------------- bundle

@cli.command("bundle")
@click.argument("ir_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for the self-contained audit bundle.")
def cmd_bundle(ir_file: Path, out_dir: Path) -> None:
    """Bundle a self-contained audit artifact (folder)."""
    ir = load_json(ir_file)
    bundle_artifact(ir, out_dir)
    click.echo(f"[+] Wrote audit bundle: {out_dir}")


# ----------------------------------------------------------------- calibrate

@cli.command("calibrate")
@click.option("--task", type=click.Choice(["role", "fact_kind", "synthetic"]), default="synthetic",
              help="Which probe to train. 'synthetic' uses a built-in toy dataset.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output checkpoint path (.pt).")
@click.option("--epochs", type=int, default=3, help="Training epochs.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--use-vib/--no-vib", default=True, help="Variational information bottleneck on/off.")
@click.option("--device", type=str, default="cpu", help="cpu or cuda.")
@click.option("--labse-model", type=str, default="sentence-transformers/LaBSE",
              help="Backbone model id.")
def cmd_calibrate(
    task: str,
    out_path: Path,
    epochs: int,
    lr: float,
    use_vib: bool,
    device: str,
    labse_model: str,
) -> None:
    """Train a probe head on the frozen LaBSE backbone.

    For real corpora, extend `calibration/dataset.py` to load your
    corrected-IR corpus. The `synthetic` task verifies the loop runs
    end-to-end on a deterministic toy dataset.
    """
    try:
        from erisml_compiler.calibration import CalibrationConfig, train_probe
        from erisml_compiler.calibration.dataset import synthetic_dataset
        from erisml_compiler.calibration.train import save_checkpoint
    except ImportError as exc:
        click.echo(f"[!] Calibration requires the [calibration] extra: {exc}", err=True)
        raise SystemExit(2)

    if task == "synthetic":
        dataset = synthetic_dataset(n_samples=64, n_classes=3)
        num_classes = 3
    else:
        click.echo(f"[!] '{task}' calibration on a real corpus needs a "
                   f"dataset loader you provide in calibration/dataset.py.", err=True)
        raise SystemExit(2)

    cfg = CalibrationConfig(
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        use_vib=use_vib,
        device=device,
        labse_model=labse_model,
    )
    click.echo(f"[#] Training probe (task={task}, epochs={epochs}, device={device}, vib={use_vib})...")
    backbone, history = train_probe(dataset, cfg)
    save_checkpoint(backbone, out_path, history=history)
    click.echo(f"[+] Wrote checkpoint: {out_path}")
    for note in history.notes:
        click.echo(f"    {note}")


# ----------------------------------------------------------------- silicon-emit

@cli.command("silicon-emit")
@click.option("--out-dir", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Output directory for the emitted C++ + Makefile.")
@click.option("--em-profile", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="EM-DAG YAML profile to compile. Default: bundled.")
@click.option("--fp-total-bits", type=int, default=16, help="Fixed-point total width.")
@click.option("--fp-int-bits", type=int, default=4, help="Fixed-point integer bits.")
def cmd_silicon_emit(
    out_dir: Path,
    em_profile: Path | None,
    fp_total_bits: int,
    fp_int_bits: int,
) -> None:
    """Emit synthesizable Vitis HLS C++ for the Tier-1 evaluator.

    Produces:
      erisml_fsm.cpp       Commitment / Legitimacy / Consent FSMs
      erisml_em_dag.cpp    EM-DAG pipeline (skeleton)
      erisml_top.cpp       Top-level kernel
      Makefile             v++ build targeting U55C
    """
    from erisml_compiler.silicon.hls_emit import (
        emit_em_dag_pipeline,
        emit_fsm_cpp,
        emit_makefile,
        emit_top_module,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "erisml_fsm.cpp").write_text(emit_fsm_cpp(), encoding="utf-8")
    (out_dir / "erisml_em_dag.cpp").write_text(
        emit_em_dag_pipeline(em_profile, fp_total_bits, fp_int_bits),
        encoding="utf-8",
    )
    (out_dir / "erisml_top.cpp").write_text(emit_top_module(), encoding="utf-8")
    (out_dir / "Makefile").write_text(emit_makefile(), encoding="utf-8")
    click.echo(f"[+] Emitted silicon-target sources to {out_dir}/")
    click.echo(f"    erisml_fsm.cpp, erisml_em_dag.cpp, erisml_top.cpp, Makefile")
    click.echo(f"[*] Next step: open an NRP Coder workspace with the U55C template")
    click.echo(f"    (see docs/nrp_coder_deployment.md), copy these files in, and run 'make'.")


# ----------------------------------------------------------------- monitor

@cli.command("monitor")
@click.argument("input_text", type=str)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output JSON trace path. Default: out/monitor_trace.json")
@click.option("--source", "source_kind", type=click.Choice(["mock", "huggingface", "remote-atlas"]),
              default="mock", help="Activation source backend.")
@click.option("--model-id", type=str, default="mock-llm-base",
              help="Model id for huggingface/remote-atlas sources.")
@click.option("--hidden-dim", type=int, default=64,
              help="Hidden dim for mock source (ignored otherwise).")
@click.option("--n-layers", type=int, default=8,
              help="Layer count for mock source (ignored otherwise).")
@click.option("--ssh-host", type=str, default=None,
              help="SSH host for remote-atlas source.")
@click.option("--ssh-user", type=str, default="claude")
@click.option("--ssh-password", type=str, default=None, envvar="ATLAS_SSH_PASSWORD",
              help="SSH password (or via $ATLAS_SSH_PASSWORD).")
@click.option("--device", type=str, default="cuda:1",
              help="Device for huggingface/remote-atlas sources.")
@click.option("--seed", type=int, default=0, help="Probe seed (fresh, untrained).")
def cmd_monitor(
    input_text: str,
    out_path: Path | None,
    source_kind: str,
    model_id: str,
    hidden_dim: int,
    n_layers: int,
    ssh_host: str | None,
    ssh_user: str,
    ssh_password: str | None,
    device: str,
    seed: int,
) -> None:
    """Run the I-EIP Monitor activation lens over INPUT_TEXT.

    Emits a JSON trace with per-layer MoralVectors and an aggregated
    layerwise summary. Uses fresh (untrained) probes by default — for
    calibrated probes, pair this with `eris-compile calibrate`.
    """
    from erisml_compiler.monitor import MockActivationSource
    from erisml_compiler.monitor.ieip_monitor import IEIPMonitor

    if source_kind == "mock":
        source = MockActivationSource(hidden_dim=hidden_dim, n_layers=n_layers, model_id=model_id)
    elif source_kind == "huggingface":
        from erisml_compiler.monitor.huggingface_source import HuggingFaceActivationSource
        source = HuggingFaceActivationSource(model_id=model_id, device=device)
    elif source_kind == "remote-atlas":
        if not ssh_host:
            raise click.UsageError("--ssh-host required for remote-atlas source")
        if not ssh_password:
            raise click.UsageError("--ssh-password (or $ATLAS_SSH_PASSWORD) required")
        from erisml_compiler.monitor.remote_source import RemoteAtlasActivationSource
        source = RemoteAtlasActivationSource(
            ssh_host=ssh_host, ssh_user=ssh_user, ssh_password=ssh_password,
            model_id=model_id, device=device,
        )
    else:
        raise click.UsageError(f"Unknown source kind: {source_kind}")

    monitor = IEIPMonitor(source, seed=seed)
    trace = monitor.monitor(input_text)
    source.close()

    if out_path is None:
        out_path = Path("out") / "monitor_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    out_path.write_text(_json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
    click.echo(f"[+] Trace written: {out_path}")
    click.echo(f"    source={trace.source_name} model_id={trace.model_id} "
               f"hidden_dim={trace.hidden_dim} layers={len(trace.per_layer)}")
    click.echo(f"    trace_hash={trace.trace_hash()}")


# ----------------------------------------------------------------- delta

@cli.command("delta")
@click.argument("ir_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("monitor_trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output JSON delta-report path. Default: out/delta_report.json")
@click.option("--divergence-threshold", type=float, default=0.35)
@click.option("--direction-break-max", type=int, default=2)
def cmd_delta(
    ir_path: Path,
    monitor_trace_path: Path,
    out_path: Path | None,
    divergence_threshold: float,
    direction_break_max: int,
) -> None:
    """Compare a text-lens IR against an activation-lens monitor trace.

    IR_PATH is a CompilerIR JSON file (output of `eris-compile compile`).
    MONITOR_TRACE_PATH is a MonitorTrace JSON file (output of
    `eris-compile monitor`). Emits a delta report keyed by moral dimension
    plus failure-mode flags.
    """
    import json as _json

    from erisml_compiler.delta import compare_morals, detect_failure_modes
    from erisml_compiler.ir.schemas import MoralVector

    ir_dict = _json.loads(ir_path.read_text(encoding="utf-8"))
    trace_dict = _json.loads(monitor_trace_path.read_text(encoding="utf-8"))

    # Pull the text-lens MoralVector from the IR. We accept three shapes:
    #   1. `global_moral_vector` (synthetic test fixtures)
    #   2. `moral_vectors[0]` (real compile output)
    #   3. `timeline[0].vector` or `moral_timeline[0].vector`
    text_mv_dict = ir_dict.get("global_moral_vector")
    if not text_mv_dict:
        mvs = ir_dict.get("moral_vectors") or []
        if isinstance(mvs, list) and mvs:
            text_mv_dict = mvs[0]
    if not text_mv_dict:
        timeline = ir_dict.get("timeline") or ir_dict.get("moral_timeline") or []
        if not timeline:
            raise click.UsageError(
                "IR has none of: global_moral_vector, moral_vectors[0], timeline[0].vector."
            )
        text_mv_dict = timeline[0]["vector"]
    text_mv = MoralVector.model_validate(text_mv_dict)
    activation_mv = MoralVector.model_validate(trace_dict["aggregated"])

    delta = compare_morals(
        text_mv,
        activation_mv,
        divergence_threshold=divergence_threshold,
        direction_break_max=direction_break_max,
    )

    # Reconstruct a MonitorTrace just enough to pass to detect_failure_modes.
    # (failure_modes only reads MonitorTrace.per_layer and .trace_hash().)
    from erisml_compiler.monitor.activation_probe import LayerProbeResult
    from erisml_compiler.monitor.ieip_monitor import MonitorTrace
    import torch as _torch

    per_layer = [
        LayerProbeResult(
            layer_index=r["layer_index"],
            layer_name=r["layer_name"],
            logits=_torch.zeros(10),
            moral_vector=MoralVector.model_validate(r["moral_vector"]),
            pooled_norm=float(r["pooled_norm"]),
        )
        for r in trace_dict["per_layer"]
    ]
    trace = MonitorTrace(
        text=trace_dict["text"],
        source_name=trace_dict["source_name"],
        model_id=trace_dict["model_id"],
        hidden_dim=trace_dict["hidden_dim"],
        per_layer=per_layer,
        aggregated=activation_mv,
        activation_norms=list(trace_dict["activation_norms"]),
        layer_indices=list(trace_dict["layer_indices"]),
    )
    report = detect_failure_modes(delta=delta, trace=trace)

    payload = {
        "ir_path": str(ir_path),
        "monitor_trace_path": str(monitor_trace_path),
        "delta": delta.to_dict(),
        "failure_modes": report.to_dict(),
    }
    if out_path is None:
        out_path = Path("out") / "delta_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    click.echo(f"[+] Delta report written: {out_path}")
    click.echo(f"    divergence={delta.divergence:.4f}  "
               f"direction_breaks={delta.direction_break_count}  "
               f"flag_for_review={delta.flag_for_review}")
    if report.fired:
        click.echo(f"    failure_modes_fired: {[m.value for m in report.fired]}")
        click.echo(f"    requires_human_review=True")
    else:
        click.echo(f"    failure_modes_fired: (none)")


# ----------------------------------------------------------------- version

@cli.command("version")
def cmd_version() -> None:
    """Print compiler and schema versions."""
    click.echo(f"erisml-compiler {__version__}  (schema {__schema_version__})")


if __name__ == "__main__":
    cli()
