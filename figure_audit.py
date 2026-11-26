"""Side-effect-free readability audit shared by report and notebook figures.

Checks every drawn text/legend artist on a figure: inside the canvas, at or
above a minimum font size, and axis-aligned tick labels must not collide.
The module never touches rcParams or the backend, so importing it cannot
perturb a notebook or another figure script.
"""

import matplotlib.text as mtext
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

DEFAULT_MIN_FONT_PT = 6.0


def mm_from_pt(pt):
    return pt * 25.4 / 72.0


def text_width_mm(text, fontsize, weight="normal"):
    """Approximate rendered width of one text line, for content-driven sizing."""
    if not text.strip():
        return 0.0
    fp = FontProperties(family="sans-serif", weight=weight)
    return mm_from_pt(TextPath((0, 0), text, size=fontsize, prop=fp).get_extents().width)


def _tick_text_ids(ax):
    """Return (drawn_ids, all_ids) of tick-label Texts. Drawn means: axis
    visible and tick location inside the view interval. (Matplotlib keeps
    Text objects for out-of-view ticks and for axes hidden via axis('off');
    they never render.)"""
    drawn, all_ids = set(), set()
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        for loc, tick in zip(axis.get_ticklocs(), axis.get_major_ticks()):
            for label in (tick.label1, tick.label2):
                if label is not None:
                    all_ids.add(id(label))
        # ax.axis("off") hides the whole axis at draw time while the Axis
        # object itself still reports visible.
        if not ax.axison or not axis.get_visible():
            continue
        for loc, tick in zip(axis.get_ticklocs(), axis.get_major_ticks()):
            if lo - 1e-9 <= loc <= hi + 1e-9:
                for label in (tick.label1, tick.label2):
                    if label is not None:
                        drawn.add(id(label))
    return drawn, all_ids


def audit_figure(fig, name, min_pt=DEFAULT_MIN_FONT_PT):
    """Return a list of human-readable violations for the drawn figure."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    problems = []
    tol = 2.0  # px

    def _inside(bb, what):
        fb = fig.bbox
        if (bb.x0 < fb.x0 - tol or bb.y0 < fb.y0 - tol
                or bb.x1 > fb.x1 + tol or bb.y1 > fb.y1 + tol):
            problems.append(f"clipped outside canvas: {what}")

    drawn_ticks, every_tick = set(), set()
    for ax in fig.axes:
        d, a = _tick_text_ids(ax)
        drawn_ticks |= d
        every_tick |= a
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if id(t) in every_tick and id(t) not in drawn_ticks:
            continue  # undrawn tick label (out of view or hidden axis)
        fs = t.get_fontsize()
        if fs < min_pt - 1e-6:
            problems.append(f"font {fs:.1f}pt < {min_pt}pt: '{s[:44]}'")
        _inside(t.get_window_extent(ren), f"'{s[:44]}'")

    for ax in fig.axes:
        for axis, horizontal in ((ax.xaxis, True), (ax.yaxis, False)):
            if not axis.get_visible():
                continue
            live = [t for t in axis.get_ticklabels()
                    if id(t) in drawn_ticks and t.get_text().strip()
                    and t.get_rotation() in (0.0, 90.0, -90.0, 270.0)]
            boxes = sorted(((t, t.get_window_extent(ren)) for t in live),
                           key=lambda tb: tb[1].x0 if horizontal else tb[1].y0)
            for (t1, b1), (t2, b2) in zip(boxes, boxes[1:]):
                overlap = b2.x0 < b1.x1 - 0.5 if horizontal else b2.y0 < b1.y1 - 0.5
                if overlap:
                    problems.append(
                        f"{'x' if horizontal else 'y'} tick labels overlap: "
                        f"'{t1.get_text()[:22]}' / '{t2.get_text()[:22]}'")
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            _inside(leg.get_window_extent(ren), "axes legend")
    for leg in fig.legends:
        if leg.get_visible():
            _inside(leg.get_window_extent(ren), "figure legend")
    return problems
