"""Generate every number, table and data figure of the GHOST manuscript from results/*.json.
Usage: python make_tables.py OUTDIR   (writes OUTDIR/results_macros.tex, OUTDIR/tab_*.tex, ...)"""
import os, sys, json, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'generated')
os.makedirs(OUT, exist_ok=True)
MISSING = []


def J(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        MISSING.append(name); return None
    return json.load(open(p))


def f1(x): return f'{x:.1f}'
def f2(x): return f'{x:.2f}'
def f3(x): return f'{x:.3f}'


macros = {}


def M(name, val):
    macros[name] = val if val is not None else r'\textbf{[missing]}'


def write(name, s):
    open(os.path.join(OUT, name), 'w').write(s)


# ------------------------------------------------------------------ victims
vr = J('victim_resnet20_s0.json'); vv = J('victim_vgg11h_s0.json')
M('VicAccR', f1(vr['test_acc']) if vr else None)
M('VicAccV', f1(vv['test_acc']) if vv else None)

# ------------------------------------------------------------------ masking bound
bits = 0.5 * math.log2(1 + 1 / 100)
M('MaskBitsPerCoord', f'{bits:.4f}')
M('MaskBitsImage', f'{bits * 16 * 32 * 32:.0f}')
M('MaskBitsRhoK', f'{0.5 * math.log2(1 + 1e-6) * 16 * 32 * 32:.3f}')

# ------------------------------------------------------------------ exactness
rows, maxrel = [], 0.0
for arch, lab in [('resnet20', 'ResNet-20'), ('vgg11h', 'VGG-11')]:
    ex = J(f'exact_{arch}.json')
    if not ex: continue
    for r in ex:
        maxrel = max(maxrel, r['rel_logit_err'])
        rows.append(f"{lab} & {r['rho']:g} & {f2(r['base_acc'])} & {f2(r['ghost_acc'])} & {f2(r['agreement'])} & "
                    f"{r['rel_logit_err']:.1e} \\\\")
agr = {}
for arch in ['resnet20', 'vgg11h']:
    ex = J(f'exact_{arch}.json') or []
    for r in ex: agr.setdefault(r['rho'], []).append(r['agreement'])
M('ExMinAgreeK', f'{min(agr.get(1000, [100])):.2f}'); M('ExMinAgreeTenK', f'{min(agr.get(10000, [100])):.2f}')
M('ExMaxRelErrHundred', f"{max(r['rel_logit_err'] for a in ['resnet20','vgg11h'] for r in (J(f'exact_{a}.json') or []) if r['rho'] <= 100):.1e}".replace('e-0', r'\times10^{-') + '}')
M('ExMaxRelErr', f'{maxrel:.1e}'.replace('e-0', r'\times10^{-').replace('e-', r'\times10^{-') + '}' if rows else None)
write('tab_exact.tex', r"""\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrcccc@{}}
\toprule
Model & $\rho$ & Orig.\ (\%) & GHOST (\%) & Agree.\ (\%) & Logit err. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ overhead
rows = []
for arch, lab in [('resnet20', 'ResNet-20'), ('vgg11h', 'VGG-11')]:
    ov = J(f'overhead_{arch}.json'); mc = J(f'macs_{arch}.json')
    if not ov or not mc: continue
    for r, m in zip(ov, mc):
        rows.append(f"{lab} & {r['n_edge_convs']} ({r.get('n_outsourced', r['n_edge_convs'])}) & {f1(m['M_orig_edge'])} & "
                    f"{f1(m['M_tee_first'] + m['M_tee_online'])} & {f1(m['M_tee_offline'])} & {f1(m['M_ree'])} & "
                    f"{f2(r['tee_online_ms'])} & {f2(r['ree_ms'])} & {f2(r['aes_ms'])} & {f2(r['tee_offline_ms'])} & "
                    f"{f1(r['kb_tee_to_ree'] + r['kb_ree_to_tee'])} & {f1(r['kb_to_cloud'])} \\\\")
    M('OrigMs' + ('R' if arch == 'resnet20' else 'V'), f2(ov[0]['plain_ms']))
write('tab_overhead.tex', r"""\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
& Edge convs & \multicolumn{4}{c}{Multiply-accumulates per request (M)} & \multicolumn{4}{c}{Measured time per request (ms)} & \multicolumn{2}{c}{Data (KB)} \\\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}
Model & (outsourced) & Original edge & TEE online & TEE offline & REE & TEE online & REE & AES-GCM & TEE offline & REE--TEE & to cloud \\\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ model stealing
LAB = {
    'scratch': ('Scratch', '--'),
    'public': ('Checkpoint, fine-tuned', 'ckpt.'),
    'plain': ('Plaintext edge weights', '--'),
    'ghost': ('GHOST tensors, direct', '--'),
    'arrow_own': ('GHOST, direction matching', 'own'),
    'unmix_own': ('GHOST, LS un-mixing', 'own'),
    'arrow_oracle': ('GHOST, direction matching', 'indep.'),
    'unmix_oracle': ('GHOST, LS un-mixing', 'indep.'),
    'projK_oracle': ('GHOST, projection, $B$ given', 'indep.'),
    'ps_arrow_oracle': ('P\&S ablation, direction matching', 'indep.'),
    'left_projK_oracle': ('Left-only ablation, projection', 'indep.'),
}
GHOST_ATTACKS = ['ghost', 'arrow_own', 'unmix_own', 'arrow_oracle', 'unmix_oracle']   # attacks without oracle knowledge of B


def steal(tag, arch, n, cond):
    return J(f'steal{tag}_{arch}_n{n}_{cond}.json')


def werr_fmt(r):
    w = r.get('w_rel_err')
    if w is None: return '--'
    return '$>10$' if w > 10 else f'{w:.3f}'


def steal_table(tag, conds, refname):
    out, refs = [], []
    for c in conds:
        r = steal(tag, 'resnet20', 5000, c)
        if r is None: continue
        lab, ref = LAB[c]
        if ref == 'indep.': ref = refname
        if r.get('ref_rel_err'): refs.append(r['ref_rel_err'])
        sep = r'\midrule' + '\n' if c in ('ghost', 'ps_arrow_oracle') else ''
        out.append(f"{sep}{lab} & {ref} & {werr_fmt(r)} & {f1(r['acc'])} \\\\")
    note = f"Weight error of the reference model alone: {np.mean(refs):.3f}." if refs else ''
    return r"""\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcc@{}}
\toprule
Starting point & Ref. & Weight err. & Acc.\ (\%) \\
\midrule
""" + '\n'.join(out) + r"""
\bottomrule
\multicolumn{4}{@{}l@{}}{\footnotesize """ + note + r"""}
\end{tabular}"""


main = ['scratch', 'plain', 'ghost', 'arrow_own', 'unmix_own', 'arrow_oracle', 'unmix_oracle', 'projK_oracle',
        'ps_arrow_oracle', 'left_projK_oracle']
write('tab_steal.tex', steal_table('', main, 'indep.'))
g = lambda c, t='': (steal(t, 'resnet20', 5000, c) or {}).get('acc')
M('StealScratchR', f1(g('scratch')) if g('scratch') is not None else None)
M('StealPlainR', f1(g('plain')) if g('plain') is not None else None)
M('StealGhostR', f1(g('ghost')) if g('ghost') is not None else None)
ga = [g(c) for c in GHOST_ATTACKS if g(c) is not None]
M('StealBestGhostR', f1(max(ga)) if ga else None)
M('StealGapR', f1(max(0.0, max(ga) - g('scratch'))) if ga and g('scratch') is not None else None)
M('StealPsArrowR', f1(g('ps_arrow_oracle')) if g('ps_arrow_oracle') is not None else None)
M('StealLeftProjR', f1(g('left_projK_oracle')) if g('left_projK_oracle') is not None else None)

pub = ['public', 'plain', 'ghost', 'arrow_oracle', 'unmix_oracle', 'projK_oracle', 'ps_arrow_oracle', 'left_projK_oracle']
write('tab_public.tex', steal_table('pub', pub, 'ckpt.'))
M('PubPublic', f1(g('public', 'pub')) if g('public', 'pub') is not None else None)
M('PubPlain', f1(g('plain', 'pub')) if g('plain', 'pub') is not None else None)
M('PubPsArrow', f1(g('ps_arrow_oracle', 'pub')) if g('ps_arrow_oracle', 'pub') is not None else None)
pa = [g(c, 'pub') for c in ['ghost', 'arrow_oracle', 'unmix_oracle', 'projK_oracle'] if g(c, 'pub') is not None]
M('PubBestGhost', f1(max(pa)) if pa else None)
gv = lambda c: (J(f'steal_vgg11h_n5000_{c}.json') or {}).get('acc')
for key, c in [('StealScratchV', 'scratch'), ('StealPlainV', 'plain'), ('StealGhostV', 'ghost'), ('StealArrowV', 'arrow_own')]:
    v = gv(c); M(key, None if v is None else f1(v))
ge = lambda c: (steal('pub', 'resnet20', 5000, c) or {}).get('w_rel_err')
for key, c in [('PubPsArrowErr', 'ps_arrow_oracle'), ('PubLeftProjErr', 'left_projK_oracle'), ('PubArrowErr', 'arrow_oracle'),
               ('PubUnmixErr', 'unmix_oracle'), ('PubProjKErr', 'projK_oracle')]:
    v = ge(c); M(key, None if v is None else ('$>10$' if v > 10 else f3(v)))
vft = J('victimft_resnet20.json')
M('VicAccFt', f1(vft['test_acc']) if vft else None)

# accuracy against adversary data size
series = {'scratch': 'Scratch', 'plain': 'Plaintext edge weights', 'best': 'Best attack on GHOST', 'oracle': 'Projection, $B$ given'}
pts = {k: [] for k in series}
for n in [1000, 5000, 10000]:
    r = steal('', 'resnet20', n, 'projK_oracle')
    if r: pts['oracle'].append((n, r['acc']))
for n in [1000, 5000, 10000]:
    for k in ['scratch', 'plain']:
        r = steal('', 'resnet20', n, k)
        if r: pts[k].append((n, r['acc']))
    accs = [steal('', 'resnet20', n, c)['acc'] for c in GHOST_ATTACKS if steal('', 'resnet20', n, c)]
    if accs: pts['best'].append((n, max(accs)))
style = {'scratch': 'navy, mark=o, dashed', 'plain': 'alert, mark=square*', 'best': 'gold!90!black, mark=triangle*, very thick', 'oracle': 'sky!80!black, mark=diamond*, dotted'}
plots = ''
for k, lab in series.items():
    if pts[k]:
        plots += f"\\addplot[{style[k]}, line width=1pt] coordinates {{{' '.join(f'({n},{a:.2f})' for n, a in pts[k])}}};\n\\addlegendentry{{{lab}}}\n"
write('fig_steal_n.tex', r"""\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=0.62\columnwidth, xmode=log, xtick={1000,5000,10000},
  xticklabels={1k,5k,10k}, xlabel={Adversary images $n$}, ylabel={Surrogate accuracy (\%)},
  legend style={font=\scriptsize, at={(0.5,-0.32)}, anchor=north, draw=navy, legend columns=2},
  tick label style={font=\scriptsize}, label style={font=\footnotesize}, axis line style={navy},
  ymajorgrids, grid style={navy!25}]
""" + plots + r"""\end{axis}
\end{tikzpicture}""")

# ------------------------------------------------------------------ inversion
inv = J('inversion_resnet20.json')
rows = []
if inv:
    for r in inv:
        where = 'First outsourced conv.' if r['depth'] == 1 else 'Last outsourced conv.'
        rows.append(f"{where} & {r['rho']:g} & {f2(r['psnr'])} & {f3(r['ssim'])} \\\\")
    fz = [r for r in inv if r['depth'] == 1 and r['rho'] == 0]; ft = [r for r in inv if r['depth'] == 1 and r['rho'] == 10]
    M('InvPsnrFirstZero', f1(fz[0]['psnr']) if fz else None)
    M('InvPsnrFirstTen', f1(ft[0]['psnr']) if ft else None)
else:
    M('InvPsnrFirstZero', None); M('InvPsnrFirstTen', None)
base = J('inversion_baseline.json')
M('InvPsnrMean', f1(base['psnr_mean_image']) if base else None)
if base: rows.append(r'\midrule' + f"\nMean image of $\\mathcal{{D}}_{{\\mathcal{{A}}}}$ & -- & {f2(base['psnr_mean_image'])} & {f3(base['ssim_mean_image'])} \\\\")
write('tab_inversion.tex', r"""\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Observed operand & $\rho$ & PSNR (dB) & SSIM \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# reconstruction examples (first edge convolution)
try:
    import torch, matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for rho in (0, 10):
        cands = sorted(glob.glob(os.path.join(R, f'inv_examples_resnet20_*_rho{rho}.pt')))
        first = [c for c in cands if 'blocks.0.conv1' in c or c.endswith(f'conv1_rho{rho}.pt')]
        if not first: MISSING.append(f'inv_examples_resnet20_<first>_rho{rho}.pt'); continue
        p = first[0]
        d = torch.load(p)
        fig, ax = plt.subplots(2, 4, figsize=(3.4, 1.8))
        for i in range(4):
            ax[0, i].imshow(d['orig'][i].permute(1, 2, 0).numpy()); ax[1, i].imshow(d['rec'][i].permute(1, 2, 0).numpy())
        for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
        for a in ax.ravel():
            for sp in a.spines.values(): sp.set_color('#011E4B')
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.04, 0.04)
        fig.savefig(os.path.join(OUT, f'inv_rho{rho}.pdf')); plt.close(fig)
except Exception as e:
    print('example figure failed:', e)

# ------------------------------------------------------------------ extraction
ext = J('extract_resnet20.json')
plots = ''
if ext:
    for pol, lab, st in [('probs', 'Probability vector', 'navy, mark=*'), ('label', 'Label only', 'alert, mark=square*')]:
        p = sorted((r['budget'], r['acc']) for r in ext if r['policy'] == pol)
        plots += f"\\addplot[{st}, line width=1pt] coordinates {{{' '.join(f'({b},{a:.2f})' for b, a in p)}}};\n\\addlegendentry{{{lab}}}\n"
    if vr:
        plots += f"\\addplot[gold!90!black, dashed, line width=1pt, domain=800:12000, samples=2] {{{vr['test_acc']:.2f}}};\n\\addlegendentry{{Victim}}\n"
write('fig_extract.tex', r"""\begin{tikzpicture}
\begin{axis}[width=\columnwidth, height=0.62\columnwidth, xmode=log, xtick={1000,5000,10000},
  xticklabels={1k,5k,10k}, xlabel={Authorised queries $Q$}, ylabel={Surrogate accuracy (\%)},
  legend style={font=\scriptsize, at={(0.5,-0.32)}, anchor=north, draw=navy, legend columns=2},
  tick label style={font=\scriptsize}, label style={font=\footnotesize}, axis line style={navy},
  ymajorgrids, grid style={navy!25}, xmin=800, xmax=12000]
""" + plots + r"""\end{axis}
\end{tikzpicture}""")

# ------------------------------------------------------------------ MIA
mia = J('mia_resnet20.json')
rows = []
PN = {'probs': 'Probability vector', 'top1_rounded': 'Label + rounded conf.', 'label': 'Label only'}
if mia:
    for pol in ['probs', 'top1_rounded', 'label']:
        for eng in ['plain', 'ghost']:
            r = [x for x in mia if x['policy'] == pol and x['engine'] == eng][0]
            rows.append(f"{PN[pol] if eng == 'plain' else ''} & {'Original' if eng == 'plain' else 'GHOST'} & "
                        f"{f3(r['auc_loss'])} & {f3(r['auc_confidence'])} & {f3(r['auc_mentropy'])} & {f3(r['auc_correctness'])} \\\\")
write('tab_mia.tex', r"""\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Release policy & Model & Loss & Conf. & Mod.\ entropy & Correctness \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ token gate
tok = J('token_tests.json')
rows = []
if tok:
    for r in tok:
        rows.append(f"{r['scenario'].replace('Direct REE execution without authorisation', 'Protected pipeline run without authorisation')} & {'Served' if r['outcome'] == 'served' else 'Refused'} \\\\")
write('tab_token.tex', r"""\footnotesize
\begin{tabular}{@{}lc@{}}
\toprule
Request & TEE response \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ blind identification of B
ica = J('blind_icapub2_resnet20_L3.json')
rows = []
if ica:
    for r in ica:
        rows.append(f"{r['rho']:g} & {r['repeats']} & {r['n_inputs']} & {f2(r['alignment'])} & "
                    f"{f3(r['err_projection_estimated_B'])} & {f3(r['err_projection_estimated_B_refscaled'])} \\\\")
    M('IcaRefErr', f3(ica[0]['err_reference'])); M('IcaTrueB', f3(ica[0]['err_projection_true_B']))
    M('IcaBestErr', f3(min(r['err_projection_estimated_B_refscaled'] for r in ica)))
    M('IcaCin', str(ica[0]['c_in']))
else:
    for k in ['IcaRefErr', 'IcaTrueB', 'IcaBestErr', 'IcaCin']: M(k, None)
write('tab_ica.tex', r"""\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}cccccc@{}}
\toprule
$\rho$ & Repeats $m$ & Inputs & Alignment & \multicolumn{2}{c}{Weight error after projection} \\
\cmidrule(l){5-6}
& & & & ICA estimate & with reference scales \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ macros
write('results_macros.tex', '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros.items()) + '\n')
print('missing result files:', sorted(set(MISSING)))
print('macros without values:', [k for k, v in macros.items() if 'missing' in v])
