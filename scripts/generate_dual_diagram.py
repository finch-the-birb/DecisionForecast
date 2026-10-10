"""Generate publication-grade architecture diagram for c1_dual (Dual-Token TimeXer).

Generates:
1. Articles/figures/c1_dual_architecture.drawio (editable Draw.io XML with math support)
2. Articles/figures/c1_dual_architecture.svg (standalone crisp vector graphic)

Key architectural elements:
- Endogenous Price Patches + rFFT [B, K=9, d_model=64]
- Two isolated global tokens: G_ts [B, 1, 64] and G_text [B, 1, 64]
- Dual Attention Mask [11 x 11] preventing G_ts <-> G_text interaction
- Independent parallel Cross-Attentions to indicators and text
- Joint Masked Self-Attention
- Dual Gated Token Bridge with tanh(alpha_ts) and tanh(alpha_text)
"""

from __future__ import annotations

import html
from pathlib import Path


def create_drawio_xml() -> str:
    elements = []
    cell_id = 2

    def next_id() -> str:
        nonlocal cell_id
        cid = f"cell_{cell_id}"
        cell_id += 1
        return cid

    def add_box(
        val: str,
        x: int,
        y: int,
        w: int,
        h: int,
        fill: str,
        stroke: str,
        font_color: str = "#1A202C",
        font_size: int = 11,
        bold: bool = False,
        rounded: int = 1,
        dashed: int = 0,
        align: str = "center",
        valign: str = "middle",
        custom_style: str = "",
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=8 4;" if dashed else "dashed=0;"
        b_str = "fontStyle=1;" if bold else ""
        escaped_val = html.escape(val).replace("\n", "&#xa;")
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=1.5;fontFamily=Helvetica,Arial,sans-serif;fontSize={font_size};"
            f"fontColor={font_color};align={align};verticalAlign={valign};{dash_str}{b_str}{custom_style}"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_val}" style="{style}" vertex="1" parent="1">\n'
            f'  <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    def add_edge(
        src: str,
        tgt: str,
        label: str = "",
        stroke: str = "#4A5568",
        stroke_width: float = 1.5,
        exit_pt: tuple[float, float] = (0.5, 0.0),
        entry_pt: tuple[float, float] = (0.5, 1.0),
        dashed: int = 0,
        edge_style: str = "orthogonalEdgeStyle",
    ) -> str:
        cid = next_id()
        dash_str = "dashed=1;dashPattern=6 3;" if dashed else "dashed=0;"
        escaped_label = html.escape(label).replace("\n", "&#xa;")
        style = (
            f"edgeStyle={edge_style};rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={stroke_width};fontSize=10;fontColor=#2D3748;"
            f"fontFamily=Helvetica,Arial,sans-serif;labelBackgroundColor=#FFFFFF;{dash_str}"
            f"exitX={exit_pt[0]};exitY={exit_pt[1]};exitDx=0;exitDy=0;"
            f"entryX={entry_pt[0]};entryY={entry_pt[1]};entryDx=0;entryDy=0;"
        )
        xml = (
            f'<mxCell id="{cid}" value="{escaped_label}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">\n'
            f'  <mxGeometry relative="1" as="geometry" />\n'
            f"</mxCell>"
        )
        elements.append(xml)
        return cid

    # Top Header & Banners
    add_box(
        "TimeXer-Dual Architecture: Modality Isolation via Dual Global Tokens",
        50, 30, 1500, 48,
        fill="#2B6CB0", stroke="#2B6CB0", font_color="#FFFFFF", font_size=18, bold=True, rounded=1
    )
    add_box(
        "Dual-Token TimeXer: G_ts (Indicators) and G_text (Sentiment) with Additive Attention Mask Isolation",
        50, 85, 1500, 28,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#4A5568", font_size=12, bold=False, rounded=1
    )

    # Legend
    leg_y = 125
    add_box("MODALITY COLOR CODE:", 50, leg_y, 160, 32, fill="none", stroke="none", font_color="#2D3748", font_size=11, bold=True, align="left")
    add_box("Endogenous Price (OHLCV / FFT)", 220, leg_y, 230, 32, fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True)
    add_box("Exogenous Indicators (TreeSHAP 25)", 465, leg_y, 250, 32, fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True)
    add_box("Exogenous Text (FinBERT 15D)", 730, leg_y, 230, 32, fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=11, bold=True)
    add_box("Masked Attention & Dual Bridge", 975, leg_y, 230, 32, fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=11, bold=True)
    add_box("Trainable Tokens (G_ts, G_text)", 1220, leg_y, 230, 32, fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=11, bold=True)
    add_box("Prediction Head", 1465, leg_y, 85, 32, fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True)

    # =========================================================================
    # MAIN COLUMN: Flow of c1_dual (Left side, x = 50 to 950)
    # =========================================================================
    mx_x = 50

    # Top: Forecast & Head
    pred_node = add_box(
        "Forecast Horizon Output\n$$\\hat{Y} \\in \\mathbb{R}^{B \\times 7}$$\n(Price Forecast Horizon $H=7$)",
        mx_x + 350, 180, 250, 55,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11, bold=True
    )
    head_node = add_box(
        "Forecast Head: Linear Projection\n$$\\text{Linear}(d_{\\text{model}}=64 \\to H=7)$$",
        mx_x + 350, 260, 250, 50,
        fill="#EBEDEF", stroke="#2C3E50", font_color="#1B2631", font_size=11
    )
    pool_node = add_box(
        "Patch Pooling (Last Patch Token)\n$$Z_{\\text{pool}} = Z_{\\text{patch}}[:, -1, :] \\in \\mathbb{R}^{B \\times 64}$$",
        mx_x + 325, 335, 300, 50,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11
    )

    # Dual Encoder Layer Container
    add_box(
        "Dual-Token TimeXer Layer (_DualEncoderLayer, e_layers=1 or 2)\n"
        "Patches interact with both G_ts and G_text; Global tokens are strictly isolated from each other",
        mx_x + 20, 410, 910, 480,
        fill="#FDFEFE", stroke="#2B6CB0", font_color="#1A365D", font_size=12, bold=True, dashed=1, align="left", valign="top"
    )

    bridge_node = add_box(
        "Gated Dual Token Bridge (_GatedGlobalToPatch)\n"
        "$$\\Delta_{\\text{ts}} = \\text{CrossAttn}(Z_{\\text{patch}}, G_{\\text{ts}}, G_{\\text{ts}}), \\quad "
        "\\Delta_{\\text{text}} = \\text{CrossAttn}(Z_{\\text{patch}}, G_{\\text{text}}, G_{\\text{text}})$$\n"
        "$$Z_{\\text{patch}} \\leftarrow \\text{LayerNorm}\\left(Z_{\\text{patch}} + \\tanh(\\alpha_{\\text{ts}}) \\Delta_{\\text{ts}} + \\tanh(\\alpha_{\\text{text}}) \\Delta_{\\text{text}}\\right)$$\n"
        "$$[B, 9, 64]$$",
        mx_x + 50, 465, 850, 75,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10, bold=True
    )

    ffn_node = add_box(
        "Feed-Forward Network & LayerNorm\n"
        "$$\\text{FFN}([Z_{\\text{patch}}; G_{\\text{ts}}; G_{\\text{text}}]) \\in \\mathbb{R}^{B \\times 11 \\times 64}$$",
        mx_x + 50, 560, 850, 45,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10
    )

    self_node = add_box(
        "Joint Masked Self-Attention (With Dual Isolation Mask)\n"
        "$$\\text{SelfAttn}\\left(Q=[P; G_{\\text{ts}}; G_{\\text{text}}], K=[P; G_{\\text{ts}}; G_{\\text{text}}], V=[P; G_{\\text{ts}}; G_{\\text{text}}], \\text{mask}=M\\right)$$\n"
        "Patches query both tokens; Cross-token attention between $G_{\\text{ts}}$ and $G_{\\text{text}}$ is masked with $-\\infty$\n"
        "$$[B, 11, 64]$$",
        mx_x + 50, 625, 850, 65,
        fill="#FEF5E7", stroke="#D35400", font_color="#7E5109", font_size=10, bold=True
    )

    concat_tokens = add_box(
        "Token Concatenation: $[Z_{\\text{patch}}; G_{\\text{ts}}; G_{\\text{text}}] \\in \\mathbb{R}^{B \\times 11 \\times 64}$",
        mx_x + 50, 710, 850, 40,
        fill="#EDF2F7", stroke="#CBD5E0", font_color="#2D3748", font_size=11
    )

    # Parallel Cross-Attentions
    cross_ts = add_box(
        "Technical Cross-Attention\n"
        "$$G_{\\text{ts}} \\leftarrow \\text{LN}(G_{\\text{ts}} + \\text{CrossAttn}(G_{\\text{ts}}, TS_{\\text{exo}}))$$\n"
        "$$[B, 1, 64]$$",
        mx_x + 350, 775, 260, 55,
        fill="#FEF5E7", stroke="#27AE60", font_color="#145A32", font_size=10
    )
    cross_text = add_box(
        "Text Cross-Attention\n"
        "$$G_{\\text{text}} \\leftarrow \\text{LN}(G_{\\text{text}} + \\text{CrossAttn}(G_{\\text{text}}, \\text{Text}_{\\text{exo}}))$$\n"
        "$$[B, 1, 64]$$",
        mx_x + 640, 775, 260, 55,
        fill="#FEF5E7", stroke="#8E44AD", font_color="#4A235A", font_size=10
    )
    patch_pass = add_box(
        "Price Patches Passage\n$$Z_{\\text{patch}} \\in \\mathbb{R}^{B \\times 9 \\times 64}$$\n(Unmodified prior to Self-Attn)",
        mx_x + 50, 775, 270, 55,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )

    # Trainable Tokens Initialized
    g_ts_init = add_box(
        "Trainable G_ts\n$$\\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        mx_x + 350, 855, 120, 45,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=10, bold=True
    )
    g_text_init = add_box(
        "Trainable G_text\n$$\\text{nn.Parameter}$$\n$$[B, 1, 64]$$",
        mx_x + 640, 855, 120, 45,
        fill="#FDEBD0", stroke="#B9770E", font_color="#7D6608", font_size=10, bold=True
    )

    # Embeddings & Projections
    patch_proj = add_box(
        "Patch Projection + PosEmb\n$$\\text{Linear}(24+6 \\to 64) + E_{\\text{pos}}$$\n$$Z_{\\text{patch}} \\in \\mathbb{R}^{B \\times 9 \\times 64}$$",
        mx_x + 50, 930, 270, 55,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    unfold_box = add_box(
        "Unfold (P=12, S=6) + Close rFFT\n$$2 \\times 12 = 24 \\text{ values} + 6 \\text{ FFT bins}$$\n$$[B, 9, 30]$$",
        mx_x + 50, 1010, 270, 55,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=10
    )
    ts_proj = add_box(
        "Indicator Projection\n$$\\text{Linear}(25 \\to 64)$$\n$$TS_{\\text{exo}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        mx_x + 480, 960, 200, 55,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=10
    )
    text_proj = add_box(
        "Text Projection\n$$\\text{Linear}(15 \\to 64)$$\n$$\\text{Text}_{\\text{exo}} \\in \\mathbb{R}^{B \\times 60 \\times 64}$$",
        mx_x + 770, 960, 160, 55,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=10
    )

    # Raw Inputs (Bottom)
    in_price = add_box(
        "Endogenous Price Input\n$$X_{\\text{price}} \\in \\mathbb{R}^{B \\times 60 \\times 2}$$\n[Close, Volume channels]",
        mx_x + 50, 1100, 270, 65,
        fill="#E8F1FC", stroke="#2B6CB0", font_color="#1A365D", font_size=11, bold=True
    )
    in_ts = add_box(
        "Exogenous Technical Series\n$$X_{\\text{ts}} \\in \\mathbb{R}^{B \\times 60 \\times 25}$$\n(TreeSHAP 25 Technical Indicators)",
        mx_x + 480, 1100, 200, 65,
        fill="#EAFBF1", stroke="#27AE60", font_color="#145A32", font_size=11, bold=True
    )
    in_text = add_box(
        "Exogenous Text Sentiment\n$$X_{\\text{text}} \\in \\mathbb{R}^{B \\times 60 \\times 15}$$\n(FinBERT Compact Sentiment Scores)",
        mx_x + 770, 1100, 160, 65,
        fill="#F5EEF8", stroke="#8E44AD", font_color="#4A235A", font_size=11, bold=True
    )

    # Connections in Main Column
    add_edge(in_price, unfold_box, "[B, 60, 2]")
    add_edge(unfold_box, patch_proj, "[B, 9, 30]")
    add_edge(patch_proj, patch_pass, "[B, 9, 64]")
    add_edge(in_ts, ts_proj, "[B, 60, 25]")
    add_edge(in_text, text_proj, "[B, 60, 15]")
    add_edge(g_ts_init, cross_ts, "[B, 1, 64]")
    add_edge(ts_proj, cross_ts, "[B, 60, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.8, 1.0))
    add_edge(g_text_init, cross_text, "[B, 1, 64]")
    add_edge(text_proj, cross_text, "[B, 60, 64]", exit_pt=(0.5, 0.0), entry_pt=(0.8, 1.0))

    add_edge(patch_pass, concat_tokens, "Patches [B, 9, 64]")
    add_edge(cross_ts, concat_tokens, "G_ts [B, 1, 64]")
    add_edge(cross_text, concat_tokens, "G_text [B, 1, 64]")

    add_edge(concat_tokens, self_node, "[B, 11, 64]")
    add_edge(self_node, ffn_node, "[B, 11, 64]")
    add_edge(ffn_node, bridge_node, "[B, 11, 64]")
    add_edge(bridge_node, pool_node, "Z_patch [B, 9, 64]")
    add_edge(pool_node, head_node, "[B, 64]")
    add_edge(head_node, pred_node, "[B, 7]")


    # =========================================================================
    # RIGHT COLUMN: Architectural Insets (x = 980 to 1520)
    # =========================================================================
    rx_x = 980
    rx_w = 540

    # Inset 1: The Dual Attention Isolation Mask Matrix
    add_box(
        "Mathematical Mechanism (1): Dual Isolation Mask",
        rx_x, 180, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )
    add_box(
        "Attention Mask Matrix $M \\in \\mathbb{R}^{11 \\times 11}$:\n\n"
        "                  Patches (0..8)   |   G_ts (9)   |  G_text (10)\n"
        "Patches (0..8) :       0           |      0       |      0\n"
        "G_ts     (9)   :       0           |      0       |    -∞ (BLOCKED)\n"
        "G_text  (10)   :       0           |    -∞ (BLOCKED) |      0\n\n"
        "• Patches attend freely to both exogenous modalities ($G_{\\text{ts}}$ and $G_{\\text{text}}$).\n"
        "• $G_{\\text{ts}}$ and $G_{\\text{text}}$ CANNOT attend to each other ($-\\infty$), preventing\n"
        "  cross-talk between noisy daily sentiment and structured technical regime indicators.",
        rx_x, 230, rx_w, 180,
        fill="#FFF5F5", stroke="#E53E3E", font_color="#742A2A", font_size=10, bold=False, align="left", valign="top"
    )

    # Inset 2: The Dual Gated Token Bridge
    add_box(
        "Mathematical Mechanism (2): Dual Gated Token Bridge",
        rx_x, 430, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )
    add_box(
        "$$\\Delta_{\\text{ts}} = \\text{MHA}\\left(\\text{Query}=Z_{\\text{patch}}, \\text{Key}=G_{\\text{ts}}, \\text{Value}=G_{\\text{ts}}\\right)$$\n"
        "$$\\Delta_{\\text{text}} = \\text{MHA}\\left(\\text{Query}=Z_{\\text{patch}}, \\text{Key}=G_{\\text{text}}, \\text{Value}=G_{\\text{text}}\\right)$$\n"
        "$$Z_{\\text{patch}}^{\\text{out}} = \\text{LayerNorm}\\left(Z_{\\text{patch}} + \\tanh(\\alpha_{\\text{ts}})\\Delta_{\\text{ts}} + \\tanh(\\alpha_{\\text{text}})\\Delta_{\\text{text}}\\right)$$\n\n"
        "• Parameters $\\alpha_{\\text{ts}}, \\alpha_{\\text{text}}$ are initialized to $0$ (residual gate closed).\n"
        "• During training, each modality opens independently at its own pace,\n"
        "  guaranteeing that the network never collapses into unstable initial gradients.",
        rx_x, 480, rx_w, 180,
        fill="#FFFBEB", stroke="#D97706", font_color="#78350F", font_size=10, bold=False, align="left", valign="top"
    )

    # Inset 3: Comparison with c1_hierarchical & c1_late_fusion
    add_box(
        "Comparative Architectural Rationale",
        rx_x, 680, rx_w, 40,
        fill="#EDF2F7", stroke="#4A5568", font_color="#1A202C", font_size=12, bold=True, rounded=1
    )
    add_box(
        "Why Dual-Token rather than Hierarchical or Late Fusion?\n\n"
        "1. Compared to c1_hierarchical (Sequential Layers):\n"
        "   c1_dual processes both modalities simultaneously in a single layer,\n"
        "   eliminating artificial order bias (e.g. why indicators before text?).\n\n"
        "2. Compared to c1_late_fusion (Late MLP Residual):\n"
        "   c1_dual allows price patches to interact with full time-series indicators\n"
        "   throughout the encoder, rather than only reading the last bar $T$.\n\n"
        "3. Test MSE Performance on Dev set (15 tickers):\n"
        "   • c1_dual (trainable tokens): 0.704\n"
        "   • c1_hierarchical: 0.702\n"
        "   • c1_late_fusion (Champion): 0.697",
        rx_x, 730, rx_w, 230,
        fill="#F0FFF4", stroke="#38A169", font_color="#22543D", font_size=10, bold=False, align="left", valign="top"
    )

    # Connect Inset 1 to Self-Attn block with dotted line
    add_edge(self_node, add_box("", rx_x, 310, 1, 1, "none", "none"), "Enforces Mask M", stroke="#E53E3E", dashed=1, exit_pt=(1.0, 0.5), entry_pt=(0.0, 0.5))

    # Connect Inset 2 to Bridge block with dotted line
    add_edge(bridge_node, add_box("", rx_x, 520, 1, 1, "none", "none"), "Dual Gate Math", stroke="#D97706", dashed=1, exit_pt=(1.0, 0.5), entry_pt=(0.0, 0.5))

    xml_header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-10-10T00:30:00.000Z" agent="Antigravity" version="24.0.0" type="device">\n'
        '  <diagram id="c1-dual-architecture" name="c1_dual Architecture">\n'
        '    <mxGraphModel dx="2000" dy="1400" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1250" math="1" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0" />\n'
        '        <mxCell id="1" parent="0" />\n'
    )
    xml_footer = "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"

    return xml_header + "\n".join(elements) + "\n" + xml_footer


def create_standalone_svg() -> str:
    svg_w = 1600
    svg_h = 1250

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        "<defs>",
        '  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#4A5568" />',
        "  </marker>",
        '  <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#27AE60" />',
        "  </marker>",
        '  <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#8E44AD" />',
        "  </marker>",
        '  <filter id="shadow" x="-3%" y="-3%" width="106%" height="106%">',
        '    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.06"/>',
        "  </filter>",
        "</defs>",
    ]

    def draw_box(x, y, w, h, fill, stroke, title, lines=None, stroke_width=1.5, rx=8, dashed=False, title_color="#1A202C", title_size=13, body_size=11, bold_title=True, align="center"):
        dash_attr = ' stroke-dasharray="6,4"' if dashed else ""
        svg_parts.append(f'  <g filter="url(#shadow)">')
        svg_parts.append(f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />')
        
        anchor = "middle" if align == "center" else "start"
        text_x = x + w // 2 if align == "center" else x + 16
        start_y = y + 20 if lines else y + h // 2 + 5
        weight = "bold" if bold_title else "600"
        svg_parts.append(f'    <text x="{text_x}" y="{start_y}" text-anchor="{anchor}" fill="{title_color}" font-size="{title_size}" font-weight="{weight}">{html.escape(title)}</text>')
        if lines:
            line_y = start_y + 17
            for line in lines:
                color = "#4A5568" if not line.startswith("[") else "#2B6CB0"
                weight_line = "bold" if line.startswith("[") or "BLOCKED" in line else "normal"
                if "BLOCKED" in line or "-inf" in line:
                    color = "#E53E3E"
                svg_parts.append(f'    <text x="{text_x}" y="{line_y}" text-anchor="{anchor}" fill="{color}" font-size="{body_size}" font-weight="{weight_line}">{html.escape(line)}</text>')
                line_y += 15
        svg_parts.append("  </g>")

    def draw_arrow(x1, y1, x2, y2, label="", color="#4A5568", marker="arrow", dashed=False):
        dash_attr = ' stroke-dasharray="4,3"' if dashed else ""
        svg_parts.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash_attr} marker-end="url(#{marker})" />')
        if label:
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2
            pad_w = len(label) * 6 + 12
            svg_parts.append(f'  <rect x="{mx - pad_w//2}" y="{my - 8}" width="{pad_w}" height="16" rx="3" fill="#FFFFFF" fill-opacity="0.9" />')
            svg_parts.append(f'  <text x="{mx}" y="{my + 4}" text-anchor="middle" fill="#2D3748" font-size="10" font-weight="600">{html.escape(label)}</text>')

    # 1. Header
    svg_parts.append('  <rect x="50" y="25" width="1500" height="48" rx="8" fill="#2B6CB0" />')
    svg_parts.append('  <text x="800" y="55" text-anchor="middle" fill="#FFFFFF" font-size="19" font-weight="bold">TimeXer-Dual Architecture: Modality Isolation via Dual Global Tokens</text>')
    
    svg_parts.append('  <rect x="50" y="80" width="1500" height="30" rx="6" fill="#EDF2F7" stroke="#CBD5E0" stroke-width="1" />')
    svg_parts.append('  <text x="800" y="100" text-anchor="middle" fill="#4A5568" font-size="12" font-weight="500">Dual-Token TimeXer: G_ts (Indicators) and G_text (Sentiment) with Additive Attention Mask Isolation</text>')

    # 2. Legend
    leg_y = 125
    svg_parts.append(f'  <text x="60" y="{leg_y + 20}" fill="#2D3748" font-size="12" font-weight="bold">MODALITY COLOR CODE:</text>')
    draw_box(220, leg_y, 230, 32, "#E8F1FC", "#2B6CB0", "Endogenous Price (OHLCV / FFT)", title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(465, leg_y, 250, 32, "#EAFBF1", "#27AE60", "Exogenous Indicators (TreeSHAP 25)", title_color="#145A32", title_size=11, bold_title=True)
    draw_box(730, leg_y, 230, 32, "#F5EEF8", "#8E44AD", "Exogenous Text (FinBERT 15D)", title_color="#4A235A", title_size=11, bold_title=True)
    draw_box(975, leg_y, 230, 32, "#FEF5E7", "#D35400", "Masked Attention & Dual Bridge", title_color="#7E5109", title_size=11, bold_title=True)
    draw_box(1220, leg_y, 230, 32, "#FDEBD0", "#B9770E", "Trainable Tokens (G_ts, G_text)", title_color="#7D6608", title_size=11, bold_title=True)

    # 3. Main Column
    mx_x = 50
    draw_box(mx_x + 350, 180, 250, 50, "#EBEDEF", "#2C3E50", "Forecast Horizon Output", ["Y_hat ∈ R^[B × 7] (Horizon H=7)", "[B, 7]"], title_color="#1B2631")
    draw_box(mx_x + 350, 260, 250, 48, "#EBEDEF", "#2C3E50", "Forecast Head: Linear Projection", ["Linear(d_model=64 → H=7)"], title_color="#1B2631")
    draw_box(mx_x + 325, 335, 300, 48, "#E8F1FC", "#2B6CB0", "Patch Pooling (Last Patch Token)", ["Z_pool = Z_patch[:, -1, :]", "[B, 64]"], title_color="#1A365D")

    # Layer Container
    draw_box(mx_x + 20, 410, 910, 480, "#FDFEFE", "#2B6CB0", "Dual-Token TimeXer Layer (_DualEncoderLayer, e_layers=1)", ["Patches interact with both G_ts and G_text; Global tokens are strictly isolated from each other"], title_color="#1A365D", title_size=12, bold_title=True, dashed=True)
    
    draw_box(mx_x + 50, 465, 850, 75, "#FEF5E7", "#D35400", "Gated Dual Token Bridge (_GatedGlobalToPatch)", [
        "Δ_ts = CrossAttn(Z_patch, G_ts),   Δ_text = CrossAttn(Z_patch, G_text)",
        "Z_patch ← LayerNorm(Z_patch + tanh(α_ts) · Δ_ts + tanh(α_text) · Δ_text)",
        "[B, 9, 64]"
    ], title_color="#7E5109", title_size=11, bold_title=True)

    draw_box(mx_x + 50, 560, 850, 45, "#FEF5E7", "#D35400", "Feed-Forward Network & LayerNorm", ["FFN([Z_patch; G_ts; G_text]) ∈ R^[B × 11 × 64]"], title_color="#7E5109", title_size=10)

    draw_box(mx_x + 50, 625, 850, 65, "#FEF5E7", "#D35400", "Joint Masked Self-Attention (With Dual Isolation Mask)", [
        "SelfAttn(Q=[P; G_ts; G_text], K=[P; G_ts; G_text], V=[P; G_ts; G_text], mask=M)",
        "Patches query both tokens; G_ts <-> G_text attention is blocked with -inf",
        "[B, 11, 64]"
    ], title_color="#7E5109", title_size=11, bold_title=True)

    draw_box(mx_x + 50, 710, 850, 40, "#EDF2F7", "#CBD5E0", "Token Concatenation", ["[Z_patch; G_ts; G_text] ∈ R^[B × 11 × 64]"], title_color="#2D3748", title_size=10)

    draw_box(mx_x + 350, 775, 260, 55, "#FEF5E7", "#27AE60", "Technical Cross-Attention", ["G_ts ← LN(G_ts + CrossAttn(G_ts, TS_exo))", "[B, 1, 64]"], title_color="#145A32", title_size=10)
    draw_box(mx_x + 640, 775, 260, 55, "#FEF5E7", "#8E44AD", "Text Cross-Attention", ["G_text ← LN(G_text + CrossAttn(G_text, Text_exo))", "[B, 1, 64]"], title_color="#4A235A", title_size=10)
    draw_box(mx_x + 50, 775, 270, 55, "#E8F1FC", "#2B6CB0", "Price Patches Passage", ["Z_patch ∈ R^[B × 9 × 64] (Preserved)", "[B, 9, 64]"], title_color="#1A365D", title_size=10)

    draw_box(mx_x + 350, 855, 120, 45, "#FDEBD0", "#B9770E", "Trainable G_ts", ["G_ts ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)
    draw_box(mx_x + 640, 855, 120, 45, "#FDEBD0", "#B9770E", "Trainable G_text", ["G_text ∈ R^[B × 1 × 64]"], title_color="#7D6608", title_size=10, bold_title=True)

    draw_box(mx_x + 50, 930, 270, 55, "#E8F1FC", "#2B6CB0", "Patch Projection + PosEmb", ["Linear(24+6 → 64) + PosEmb", "Z_patch ∈ R^[B × 9 × 64]"], title_color="#1A365D", title_size=10)
    draw_box(mx_x + 50, 1010, 270, 55, "#E8F1FC", "#2B6CB0", "Unfold (P=12, S=6) + Close rFFT", ["2 × 12 = 24 values + 6 FFT bins", "[B, 9, 30]"], title_color="#1A365D", title_size=10)
    draw_box(mx_x + 480, 960, 200, 55, "#EAFBF1", "#27AE60", "Indicator Projection", ["Linear(25 → 64)", "TS_exo ∈ R^[B × 60 × 64]"], title_color="#145A32", title_size=10)
    draw_box(mx_x + 770, 960, 160, 55, "#F5EEF8", "#8E44AD", "Text Projection", ["Linear(15 → 64)", "Text_exo ∈ R^[B × 60 × 64]"], title_color="#4A235A", title_size=10)

    draw_box(mx_x + 50, 1100, 270, 65, "#E8F1FC", "#2B6CB0", "Endogenous Price Input", ["X_price ∈ R^[B × 60 × 2]", "[Close, Volume channels]"], title_color="#1A365D", title_size=11, bold_title=True)
    draw_box(mx_x + 480, 1100, 200, 65, "#EAFBF1", "#27AE60", "Exogenous Technical Series", ["X_ts ∈ R^[B × 60 × 25]", "TreeSHAP 25 Technical Indicators"], title_color="#145A32", title_size=11, bold_title=True)
    draw_box(mx_x + 770, 1100, 160, 65, "#F5EEF8", "#8E44AD", "Exogenous Text Sentiment", ["X_text ∈ R^[B × 60 × 15]", "FinBERT Compact Scores"], title_color="#4A235A", title_size=11, bold_title=True)

    # Main Column Arrows
    draw_arrow(mx_x + 185, 1100, mx_x + 185, 1068, "[B, 60, 2]")
    draw_arrow(mx_x + 185, 1010, mx_x + 185, 988, "[B, 9, 30]")
    draw_arrow(mx_x + 185, 930, mx_x + 185, 835, "[B, 9, 64]")
    draw_arrow(mx_x + 580, 1100, mx_x + 580, 1018, "[B, 60, 25]")
    draw_arrow(mx_x + 580, 960, mx_x + 550, 835, "[B, 60, 64]", marker="arrow-green")
    draw_arrow(mx_x + 850, 1100, mx_x + 850, 1018, "[B, 60, 15]")
    draw_arrow(mx_x + 850, 960, mx_x + 820, 835, "[B, 60, 64]", marker="arrow-purple")
    draw_arrow(mx_x + 410, 855, mx_x + 410, 835)
    draw_arrow(mx_x + 700, 855, mx_x + 700, 835)

    draw_arrow(mx_x + 185, 775, mx_x + 185, 755)
    draw_arrow(mx_x + 480, 775, mx_x + 480, 755)
    draw_arrow(mx_x + 770, 775, mx_x + 770, 755)

    draw_arrow(mx_x + 475, 710, mx_x + 475, 695, "[B, 11, 64]")
    draw_arrow(mx_x + 475, 625, mx_x + 475, 610)
    draw_arrow(mx_x + 475, 560, mx_x + 475, 545)
    draw_arrow(mx_x + 475, 465, mx_x + 475, 388, "[B, 9, 64]")
    draw_arrow(mx_x + 475, 335, mx_x + 475, 310, "[B, 64]")
    draw_arrow(mx_x + 475, 260, mx_x + 475, 235, "[B, 7]")

    # 4. Right Insets
    rx_x = 980
    rx_w = 540
    draw_box(rx_x, 180, rx_w, 40, "#EDF2F7", "#4A5568", "Mathematical Mechanism (1): Dual Isolation Mask", title_size=12, bold_title=True)
    draw_box(rx_x, 230, rx_w, 185, "#FFF5F5", "#E53E3E", "Attention Mask Matrix M ∈ R^[11 × 11]", [
        "                  Patches (0..8)   |   G_ts (9)   |  G_text (10)",
        "Patches (0..8) :       0           |      0       |      0",
        "G_ts     (9)   :       0           |      0       |    -inf (BLOCKED)",
        "G_text  (10)   :       0           |    -inf (BLOCKED) |      0",
        "",
        "• Patches attend freely to both exogenous modalities (G_ts and G_text).",
        "• G_ts and G_text CANNOT attend to each other (-inf), preventing",
        "  cross-talk between noisy daily sentiment and structured technical regime indicators."
    ], body_size=10, bold_title=True, align="left")

    draw_box(rx_x, 435, rx_w, 40, "#EDF2F7", "#4A5568", "Mathematical Mechanism (2): Dual Gated Token Bridge", title_size=12, bold_title=True)
    draw_box(rx_x, 485, rx_w, 185, "#FFFBEB", "#D97706", "Independent Learnable Modality Gating", [
        "Δ_ts = MHA(Query=Z_patch, Key=G_ts, Value=G_ts)",
        "Δ_text = MHA(Query=Z_patch, Key=G_text, Value=G_text)",
        "Z_patch_out = LayerNorm(Z_patch + tanh(α_ts) · Δ_ts + tanh(α_text) · Δ_text)",
        "",
        "• Parameters α_ts, α_text are initialized to 0 (residual gate closed).",
        "• During training, each modality opens independently at its own pace,",
        "  guaranteeing that the network never collapses into unstable initial gradients."
    ], body_size=10, bold_title=True, align="left")

    draw_box(rx_x, 690, rx_w, 40, "#EDF2F7", "#4A5568", "Comparative Architectural Rationale", title_size=12, bold_title=True)
    draw_box(rx_x, 740, rx_w, 230, "#F0FFF4", "#38A169", "Why Dual-Token rather than Hierarchical or Late Fusion?", [
        "1. Compared to c1_hierarchical (Sequential Layers):",
        "   c1_dual processes both modalities simultaneously in a single layer,",
        "   eliminating artificial order bias (e.g. why indicators before text?).",
        "",
        "2. Compared to c1_late_fusion (Late MLP Residual):",
        "   c1_dual allows price patches to interact with full time-series indicators",
        "   throughout the encoder, rather than only reading the last bar T.",
        "",
        "3. Test MSE Performance on Dev set (15 tickers):",
        "   • c1_dual (trainable tokens): 0.704",
        "   • c1_hierarchical: 0.702",
        "   • c1_late_fusion (Champion): 0.697"
    ], body_size=10, bold_title=True, align="left")

    # Connectors to insets
    draw_arrow(mx_x + 900, 655, rx_x, 320, "Mask M", color="#E53E3E", dashed=True)
    draw_arrow(mx_x + 900, 502, rx_x, 575, "Dual Gates", color="#D97706", dashed=True)

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def main() -> None:
    out_dir = Path("Articles/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    drawio_path = out_dir / "c1_dual_architecture.drawio"
    svg_path = out_dir / "c1_dual_architecture.svg"

    print("Generating c1_dual Draw.io XML...")
    drawio_xml = create_drawio_xml()
    drawio_path.write_text(drawio_xml, encoding="utf-8")
    print(f"Saved Draw.io file to: {drawio_path} ({len(drawio_xml)} bytes)")

    print("Generating c1_dual standalone publication SVG...")
    svg_content = create_standalone_svg()
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Saved standalone SVG to: {svg_path} ({len(svg_content)} bytes)")


if __name__ == "__main__":
    main()
