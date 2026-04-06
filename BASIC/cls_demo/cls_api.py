"""
cls_api.py — Notebook-friendly API for CLS Learner.

Provides:
  - CLSDemo:         Interactive study / predict / reset
  - noise_curve:     σ vs accuracy plot (matplotlib)
  - show_prediction: Color-block visualization (with noisy Lab support)
"""
import sys, os, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cls_learner.agent import CLSAgent
from cls_learner.config import CLSConfig
from cls_learner.interfaces import Example
from ns_learner.ns_colors import (
    RGB_PALETTE, lab_vec, add_noise, nearest_color,
    norm_lab_to_mpl, LAB_PALETTE,
)

# ── Color visualization helpers ─────────────────────────────────

# RGB 0-255 → 0-1 for matplotlib
_MPL_COLORS = {name: tuple(c / 255 for c in rgb)
               for name, rgb in RGB_PALETTE.items()}
_MPL_COLORS['UNKNOWN'] = (0.5, 0.5, 0.5)


def _draw_color_row(ax, colors, max_len, label,
                    noisy_vecs=None, show_name=True):
    """Draw one row of color blocks.

    Parameters
    ----------
    ax : matplotlib Axes
    colors : list[str]       Color names
    max_len : int             Max row length (for x-axis)
    label : str               Row label
    noisy_vecs : list[ndarray], optional
        If provided, use these Lab vectors for block fill color
        instead of the clean palette colors. Color names are still
        shown as text labels.
    show_name : bool
        Whether to show color name text inside blocks.
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(-0.3, max_len + 0.3)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(label, fontsize=10, loc='left', fontweight='bold')

    for i, c in enumerate(colors):
        # Determine fill color
        if noisy_vecs is not None and i < len(noisy_vecs):
            rgb = norm_lab_to_mpl(noisy_vecs[i])
        else:
            rgb = _MPL_COLORS.get(c, _MPL_COLORS['UNKNOWN'])

        rect = mpatches.FancyBboxPatch(
            (i, 0.1), 0.85, 0.7, boxstyle='round,pad=0.05',
            facecolor=rgb, edgecolor='#333', linewidth=1.2)
        ax.add_patch(rect)

        if show_name:
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            txt_color = 'white' if lum < 0.5 else 'black'
            ax.text(i + 0.42, 0.45, c, ha='center', va='center',
                    fontsize=7, color=txt_color, fontweight='bold')


def show_prediction(words, predicted, expected=None, title=None,
                    noisy_vecs=None):
    """Visualize a prediction as colored blocks.

    Parameters
    ----------
    words : list[str]
        Input word sequence.
    predicted : list[str]
        Predicted output color sequence.
    expected : list[str], optional
        Ground-truth output (shown for comparison).
    title : str, optional
        Custom title.
    noisy_vecs : list[ndarray], optional
        If provided, adds a row showing the actual noisy Lab colors
        the model received during training.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    rows = [('Predicted', predicted, None)]
    if expected:
        rows.append(('Expected', expected, None))
    if noisy_vecs is not None:
        # Show what the model actually saw
        noisy_names = [nearest_color(v) for v in noisy_vecs]
        rows.append(('Model saw (noisy)', noisy_names, noisy_vecs))

    n_rows = len(rows)
    max_len = max(len(r[1]) for r in rows)
    fig_w = max(4, max_len * 1.2 + 1)

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(fig_w, n_rows * 1.2 + 0.8))
    if n_rows == 1:
        axes = [axes]

    for ax, (label, colors, vecs) in zip(axes, rows):
        _draw_color_row(ax, colors, max_len, label, noisy_vecs=vecs)

    inp_str = ' '.join(words)
    fig.suptitle(f'Input: {inp_str}', fontsize=11,
                 fontweight='bold', y=1.02)
    if title:
        fig.text(0.5, -0.02, title, ha='center', fontsize=9,
                 style='italic')

    plt.tight_layout()
    return fig


def show_colors(color_names, noisy_vecs=None, title='Colors'):
    """Show a row of color blocks (standalone).

    Parameters
    ----------
    color_names : list[str]
        Color names to show.
    noisy_vecs : list[ndarray], optional
        Normalized Lab vectors — if given, they provide the fill color
        (actual perturbed shade) while names remain as labels.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(max(4, len(color_names) * 1.2), 1.5))
    _draw_color_row(ax, color_names, len(color_names), title,
                    noisy_vecs=noisy_vecs)
    plt.tight_layout()
    return fig


# ── CLSDemo class ────────────────────────────────────────────────

# mini-SCAN token → color
_SCAN_TO_COLOR = {'1': 'BLUE', '2': 'RED', '3': 'GREEN', 'DAX': 'YELLOW'}
_COLOR_TO_SCAN = {v: k for k, v in _SCAN_TO_COLOR.items()}


def _parse_miniscan():
    """Load mini-SCAN from bundled data."""
    fp = os.path.join(_HERE, 'data', 'mini_scan', 'mini_scan.txt')
    support, query = [], []
    cur = None
    for line in open(fp, encoding='utf-8'):
        line = line.strip()
        if line == '*SUPPORT*': cur = 'support'; continue
        elif line == '*QUERY*': cur = 'query'; continue
        elif line == '*GRAMMAR*': cur = 'grammar'; continue
        if cur in ('support', 'query') and line:
            m = re.match(r'IN:\s+(.*?)\s+OUT:\s+(.*)', line)
            if m:
                inp, out = m.group(1).split(), m.group(2).split()
                colors = [_SCAN_TO_COLOR.get(t, t) for t in out]
                (support if cur == 'support' else query).append(
                    {'input': inp, 'output': colors})
    return support, query


class CLSDemo:
    """Interactive CLS Learner for Jupyter notebooks.

    Parameters
    ----------
    mode : str
        'ast' (recommended) or 'stack'.
    gauss : bool
        True = CIELAB 3D Gaussian emission.
    lab_sigma : float
        Gaussian noise σ in raw Lab units (0 = no noise).
    use_hpc : bool
        Enable hippocampal memory (default False).

    Examples
    --------
    >>> demo = CLSDemo(mode='ast', gauss=True, lab_sigma=10)
    >>> demo.study_miniscan()
    >>> demo.predict(['DAX', 'thrice'])
    """

    def __init__(self, mode='ast', gauss=True, lab_sigma=0, use_hpc=False):
        self.mode = mode
        self.gauss = gauss
        self.lab_sigma = lab_sigma
        self.use_hpc = use_hpc
        self._agent = None
        self._support = []          # [(words, colors)]
        self._noisy_support = []    # [(words, colors, lab_vecs)]
        self.reset()

    def reset(self):
        """Reset the learner (clear all learned concepts)."""
        cfg = CLSConfig(
            mode=self.mode, use_hpc=self.use_hpc, n_em=3,
            rsa_alpha=0.0, gauss=self.gauss, lab_sigma=self.lab_sigma,
        )
        self._agent = CLSAgent(cfg=cfg)
        self._agent.reset_episode()
        self._support = []
        self._noisy_support = []
        return self

    def study(self, examples):
        """Train on support examples.

        Parameters
        ----------
        examples : list of (input_words, output_colors) tuples
            e.g. [(['dax'], ['RED']), (['dax', 'fep'], ['RED', 'RED', 'RED'])]
        """
        self.reset()
        self._support = examples

        # Pre-compute the noisy Lab vectors for visualization
        for inp, out in examples:
            if self.gauss and self.lab_sigma > 0:
                vecs = []
                for c in out:
                    v = lab_vec(c)
                    v = add_noise(v, self.lab_sigma)
                    vecs.append(v)
                self._noisy_support.append((inp, out, vecs))
            else:
                clean = [lab_vec(c) if c in LAB_PALETTE else None
                         for c in out]
                self._noisy_support.append((inp, out, clean))

        exs = [Example(words=list(inp), output=list(out))
               for inp, out in examples]
        self._agent.study(exs)
        print(f"✓ Studied {len(examples)} examples "
              f"(mode={self.mode}, gauss={self.gauss}, σ={self.lab_sigma})")
        return self

    def study_miniscan(self):
        """Train on mini-SCAN (14 built-in support examples)."""
        support, _ = _parse_miniscan()
        pairs = [(ex['input'], ex['output']) for ex in support]
        return self.study(pairs)

    def predict(self, words, expected=None, show=True):
        """Predict output for a query and optionally visualize.

        Parameters
        ----------
        words : list[str]
            Input word sequence.
        expected : list[str], optional
            Expected output (for comparison visualization).
        show : bool
            If True, display matplotlib figure.

        Returns
        -------
        list[str]
            Predicted output colors.
        """
        pred = self._agent.predict(list(words))
        if pred is None:
            pred = []

        ok_str = ""
        if expected:
            ok_str = " ✓" if pred == list(expected) else " ✗"

        print(f"  {words} → {pred}{ok_str}")

        if show:
            try:
                fig = show_prediction(words, pred, expected)
                import matplotlib.pyplot as plt
                plt.show()
            except ImportError:
                pass

        return pred

    def predict_all_queries(self, show=True):
        """Predict all 10 mini-SCAN queries and show results.

        Returns
        -------
        tuple (n_correct, n_total)
        """
        _, queries = _parse_miniscan()
        correct = 0
        print(f"{'─' * 60}")
        print(f"  mode={self.mode}  gauss={self.gauss}  σ={self.lab_sigma}")
        print(f"{'─' * 60}")
        for q in queries:
            pred = self.predict(q['input'], expected=q['output'], show=show)
            if pred == q['output']:
                correct += 1
        print(f"{'─' * 60}")
        print(f"  Accuracy: {correct}/{len(queries)} "
              f"({correct * 100 / len(queries):.0f}%)")
        return correct, len(queries)

    def show_support(self, show_noisy=True):
        """Visualize all support examples in one figure.

        Shows input words → output color blocks.
        When lab_sigma > 0 and show_noisy=True, each example shows
        two rows: clean target vs what the model actually saw.
        """
        if not self._support:
            print("No support examples loaded. Call study() first.")
            return

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        has_noise = self.gauss and self.lab_sigma > 0 and show_noisy
        rows_per_ex = 2 if has_noise else 1
        n_ex = len(self._noisy_support)
        n_rows = n_ex * rows_per_ex

        max_out = max(len(out) for _, out, _ in self._noisy_support)
        fig_w = max(8, max_out * 1.3 + 5)
        fig_h = n_rows * 0.55 + 1.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(-5, max_out + 0.5)
        ax.set_ylim(-n_rows * 1.0, 1.0)
        ax.axis('off')

        y = 0
        for idx, (inp, out, vecs) in enumerate(self._noisy_support):
            # Input words (left side)
            inp_str = ' '.join(inp)
            ax.text(-0.3, y - 0.35, inp_str, ha='right', va='center',
                    fontsize=9, fontfamily='monospace')
            # Arrow
            ax.text(0.0, y - 0.35, '→', ha='center', va='center',
                    fontsize=10)

            # Output color blocks (clean)
            for i, c in enumerate(out):
                rgb = _MPL_COLORS.get(c, _MPL_COLORS['UNKNOWN'])
                rect = mpatches.FancyBboxPatch(
                    (i + 0.3, y - 0.65), 0.8, 0.6,
                    boxstyle='round,pad=0.04',
                    facecolor=rgb, edgecolor='#333', linewidth=1)
                ax.add_patch(rect)
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                tc = 'white' if lum < 0.5 else 'black'
                ax.text(i + 0.7, y - 0.35, c, ha='center', va='center',
                        fontsize=6, color=tc, fontweight='bold')

            y -= 1.0

            # Noisy row
            if has_noise and vecs:
                ax.text(-0.3, y - 0.35, '', ha='right', va='center',
                        fontsize=8)
                ax.text(0.0, y - 0.35, '≈', ha='center', va='center',
                        fontsize=10, color='#888')
                for i, v in enumerate(vecs):
                    if v is not None:
                        rgb = norm_lab_to_mpl(v)
                    else:
                        rgb = (0.5, 0.5, 0.5)
                    rect = mpatches.FancyBboxPatch(
                        (i + 0.3, y - 0.65), 0.8, 0.6,
                        boxstyle='round,pad=0.04',
                        facecolor=rgb, edgecolor='#999',
                        linewidth=0.8, linestyle='--')
                    ax.add_patch(rect)
                    name = nearest_color(v) if v is not None else '?'
                    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    tc = 'white' if lum < 0.5 else 'black'
                    ax.text(i + 0.7, y - 0.35, name, ha='center',
                            va='center', fontsize=6, color=tc, alpha=0.8)
                y -= 1.0

        title = f'Support ({n_ex} examples)'
        if has_noise:
            title += f'  —  top: clean target, bottom (dashed): model saw (σ={self.lab_sigma})'
        fig.suptitle(title, fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def __repr__(self):
        return (f"CLSDemo(mode='{self.mode}', gauss={self.gauss}, "
                f"lab_sigma={self.lab_sigma}, "
                f"support={len(self._support)} examples)")


# ── Noise curve ──────────────────────────────────────────────────

def noise_curve(sigmas=None, n_repeat=5, modes=None):
    """Run noise robustness experiment and plot results.

    Parameters
    ----------
    sigmas : list[float]
        Noise levels to test (default: [0, 5, 10, 15, 20, 25, 30, 40, 50]).
    n_repeat : int
        Repeats per sigma (default 5).
    modes : list[str]
        Decoder modes to test (default: ['ast', 'stack']).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if sigmas is None:
        sigmas = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    if modes is None:
        modes = ['ast', 'stack']

    support, queries = _parse_miniscan()
    support_pairs = [(ex['input'], ex['output']) for ex in support]

    results = {m: {s: [] for s in sigmas} for m in modes}

    total_runs = len(sigmas) * n_repeat * len(modes)
    done = 0

    print(f"Running noise curve: {len(sigmas)} σ × {n_repeat} repeats "
          f"× {len(modes)} modes = {total_runs} runs")

    for sigma in sigmas:
        for rep in range(n_repeat):
            seed = 1000 * sigma + rep
            for mode in modes:
                np.random.seed(seed)
                cfg = CLSConfig(
                    mode=mode, use_hpc=False, n_em=3, rsa_alpha=0.0,
                    gauss=True, lab_sigma=sigma,
                )
                agent = CLSAgent(cfg=cfg)
                agent.reset_episode()
                exs = [Example(words=list(inp), output=list(out))
                       for inp, out in support_pairs]
                try:
                    agent.study(exs)
                    c = sum(1 for q in queries
                            if agent.predict(q['input']) == q['output'])
                    acc = c / len(queries) * 100
                except Exception:
                    acc = 0
                results[mode][sigma].append(acc)
                done += 1

        # Progress
        means = {m: np.mean(results[m][sigma]) for m in modes}
        pct = done / total_runs * 100
        print(f"  σ={sigma:>2d}: " + "  ".join(
            f"{m}={means[m]:.0f}%" for m in modes) + f"  [{pct:.0f}%]")

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_plot = {'ast': '#2196F3', 'stack': '#FF9800'}
    markers = {'ast': 'o', 'stack': 's'}

    for mode in modes:
        means = [np.mean(results[mode][s]) for s in sigmas]
        stds = [np.std(results[mode][s]) for s in sigmas]
        ax.errorbar(sigmas, means, yerr=stds,
                    marker=markers.get(mode, 'o'),
                    color=colors_plot.get(mode, '#666'),
                    label=f'{mode.upper()} decoder',
                    capsize=4, linewidth=2, markersize=7)

    ax.set_xlabel('Noise σ (Lab units)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CLS Noise Robustness — CIELAB Gaussian Emission',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sigmas)
    plt.tight_layout()

    # ── Print markdown table ──────────────────────────────────
    print(f"\n| σ   |", end="")
    for m in modes:
        print(f" {m.upper():>8s} |", end="")
    print()
    print("| --- |" + " -------- |" * len(modes))
    for sigma in sigmas:
        print(f"| {sigma:<3d} |", end="")
        for m in modes:
            mean = np.mean(results[m][sigma])
            std = np.std(results[m][sigma])
            if std > 0:
                print(f" {mean:>4.0f}±{std:.0f}% |", end="")
            else:
                print(f" {mean:>6.0f}% |", end="")
        print()

    return fig
