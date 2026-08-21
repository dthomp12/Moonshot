from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def print_round(sim: Dict[str, Any], title: str = "Moonshot Round") -> None:
    """Pretty-print a single round."""
    table = Table(title=f"[bold]{title}[/bold]", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Batter", min_width=14)
    table.add_column("Out", justify="center", width=6)
    table.add_column("Pitches", justify="right", width=7)
    table.add_column("Added", justify="right", width=8)
    table.add_column("Running", justify="right", width=9)

    for i, step in enumerate(sim["path"], 1):
        outcome = step["outcome"]
        if outcome == "HR":
            out_str = "[bold red]HR[/bold red]"
        elif outcome in ("2B", "3B", "XBH"):
            out_str = "[yellow]XBH[/yellow]"
        elif outcome == "1B":
            out_str = "[green]1B[/green]"
        elif outcome in ("BB", "HBP"):
            out_str = "[cyan]BB[/cyan]"
        else:
            out_str = "[dim]OUT[/dim]"

        table.add_row(
            str(i),
            step.get("batter", "—"),
            out_str,
            str(step["pitches"]),
            f"+{step['mult_added']:.2f}",
            f"{step['running_mult']:.2f}x",
        )

    summary = (
        f"[bold]Final:[/bold] [green]{sim['final_multiplier']:.2f}x[/green]   "
        f"Pitches: {sim['total_pitches']}   "
        f"Batters: {sim['batters_faced']}"
    )
    console.print()
    console.print(table)
    console.print(Panel(summary, border_style="green"))
    console.print()


def print_ev_analysis(
    stats: Dict[str, Any],
    title: str = "Multiplier Reach & EV Analysis",
    plot: bool = True,
) -> None:
    """
    Show probability of reaching each multiplier + EV if you cashed there.
    Also plots both curves.
    """
    reach = stats.get("reach", {})
    targets = sorted([k for k in reach.keys() if isinstance(k, (int, float))])

    table = Table(title=f"[bold]{title}[/bold]", box=box.SIMPLE_HEAVY)
    table.add_column("Target", justify="right", style="cyan")
    table.add_column("Reach %", justify="right")
    table.add_column("EV", justify="right")

    best_ev = -999.0
    best_target = None
    probs = []
    evs = []

    for t in targets:
        p = reach[t]
        ev = reach.get(f"EV_{t}x", p * t - 1.0)
        if ev > best_ev:
            best_ev = ev
            best_target = t
        probs.append(p)
        evs.append(ev)

        ev_str = f"{ev:+.3f}"
        if ev > 0.05:
            ev_str = f"[green]{ev_str}[/green]"
        elif ev < -0.05:
            ev_str = f"[red]{ev_str}[/red]"

        table.add_row(f"{t}x", f"{p:.1%}", ev_str)

    console.print()
    console.print(table)

    if best_target is not None:
        console.print(
            Panel(
                f"Highest EV cash point: [bold green]{best_target}x[/bold green]  "
                f"(EV = {best_ev:+.3f})",
                border_style="green",
            )
        )
    console.print()

    # ----- Plot -----
    if plot and targets:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color_reach = "steelblue"
        color_ev = "darkorange"

        # Reach probability
        ax1.plot(targets, probs, color=color_reach, marker="o", linewidth=2, label="Reach Probability")
        ax1.set_xlabel("Multiplier")
        ax1.set_ylabel("Probability of Reaching", color=color_reach)
        ax1.tick_params(axis="y", labelcolor=color_reach)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        # EV on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(targets, evs, color=color_ev, marker="s", linewidth=2, label="EV")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2.set_ylabel("Expected Value (if cash at this mult)", color=color_ev)
        ax2.tick_params(axis="y", labelcolor=color_ev)

        # Mark best EV
        if best_target is not None:
            ax2.scatter([best_target], [best_ev], color="red", s=100, zorder=5, label="Best EV")

        fig.legend(loc="upper right", bbox_to_anchor=(0.90, 0.90))
        plt.title("Reach Probability & EV by Multiplier")
        fig.tight_layout()
        plt.show()


def plot_starting_batter_distributions(
    opt: Dict[str, Any],
) -> None:
    """Plot each starter's reach percentage and cash-out EV curves."""
    starts = sorted(opt["all_starts"], key=lambda row: row["starting_idx"])
    if not starts:
        return

    fig, (reach_axis, ev_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(starts)))

    for start, color in zip(starts, colors):
        results = np.asarray(start["raw"])
        targets = np.asarray(start["targets"], dtype=float)
        probabilities = np.array([np.mean(results >= target) for target in targets])
        ev = probabilities * targets - 1.0

        label = f"{start['starting_idx']}: {start['batter_name']}"
        reach_axis.plot(
            targets,
            probabilities * 100,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )
        ev_axis.plot(
            targets,
            ev,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
        )

    # Plot profit line on reach plot
    reach_axis.plot(
        targets,
        100 / targets,
        color='k',
        linewidth=1,
        alpha=0.5,
    )

    reach_axis.set_ylabel("Reach probability (%)")
    reach_axis.set_title("Percentage of simulations reaching or surpassing multiplier")
    reach_axis.grid(True, alpha=0.25)
    reach_axis.set_xscale('log')

    ev_axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    ev_axis.set_xlabel("Multiplier")
    ev_axis.set_ylabel("Cash-out EV")
    ev_axis.set_title("Expected value at each cash-out multiplier")
    ev_axis.grid(True, alpha=0.25)
    ev_axis.set_xscale('log')

    reach_axis.legend(loc="best", fontsize="small")
    fig.suptitle(
        f"Starting Batter Reach and EV vs {opt['pitcher_name']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

def print_optimal_start_results(opt: Dict[str, Any]):
    """Print lineup ordered by starting position with max EV + best multiplier."""
    
    # Header
    header = Text.from_markup(
        f"[bold green]Best starting batter:[/] [bold cyan]{opt['best_batter_name']}[/] "
        f"(index {opt['best_starting_idx']})\n"
        f"[bold]Best EV:[/] [bold yellow]{opt['best_ev']:+.3f}[/] "
        f"at [bold]{opt['best_mult']}x[/]"
    )
    console.print(Panel(header, title="Optimal Starting Batter", border_style="green"))

    # Create a lookup so we can print in original lineup order
    by_idx = {r["starting_idx"]: r for r in opt["all_starts"]}

    table = Table(title=f"Lineup by Starting Position with Pitcher: {opt['pitcher_name']}", show_header=True, header_style="bold magenta")
    table.add_column("Start Idx", justify="right")
    table.add_column("Batter", style="cyan")
    table.add_column("Best EV", justify="right")
    table.add_column("At Mult", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("P95", justify="right")

    for idx in range(len(by_idx)):
        r = by_idx[idx]
        is_best = idx == opt["best_starting_idx"]

        ev_str = f"{r['best_ev']:+.3f}"
        if is_best:
            ev_str = f"[bold green]{ev_str}[/]"
        elif r["best_ev"] > 0:
            ev_str = f"[green]{ev_str}[/]"
        else:
            ev_str = f"[red]{ev_str}[/]"

        table.add_row(
            str(idx),
            r["batter_name"],
            ev_str,
            f"{r['best_mult']}x",
            f"{r['mean']:.2f}",
            f"{r['median']:.2f}",
            f"{r['p90']:.1f}",
            f"{r['p95']:.1f}",
            style="bold" if is_best else None,
        )

    console.print(table)