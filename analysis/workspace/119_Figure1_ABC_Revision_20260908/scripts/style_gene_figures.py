"""Rebuild gene-label typography from frozen plots, without refitting analyses."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import sys

import matplotlib

def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / 'review/gene_typography_20260908'
OUT = QA / 'figures'
GENES = ['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
PATTERN = re.compile(r'\b(' + '|'.join(GENES) + r')\b')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry(fig):
    """Capture numerical artists separately from labels and font metrics."""
    arrays = []
    for ax in fig.axes:
        arrays.extend([np.asarray(ax.get_xlim()), np.asarray(ax.get_ylim())])
        for line in ax.lines:
            arrays.extend([np.asarray(line.get_xdata()), np.asarray(line.get_ydata())])
        for collection in ax.collections:
            arrays.append(np.asarray(collection.get_offsets()))
            arrays.extend(np.asarray(p.vertices) for p in collection.get_paths())
        for im in ax.images:
            arrays.append(np.asarray(im.get_array()))
        for patch in ax.patches:
            arrays.append(np.asarray(patch.get_path().vertices))
    return arrays


def save(fig, stem, records):
    fig.canvas.draw()  # Materialise tick labels before recording the text invariant.
    old_geometry = geometry(fig)
    old_text = [t.get_text() for t in fig.findobj(Text)]
    changes = []
    for text in fig.findobj(Text):
        label = text.get_text()
        if not PATTERN.search(label) or '$\\mathit{' in label:
            continue
        # Whole-gene labels/lists can use the native italic font. Counts stay roman.
        if not re.sub(r'[\s,]+', '', PATTERN.sub('', label)):
            text.set_fontstyle('italic')
        else:
            formatted = PATTERN.sub(lambda m: r'$\mathit{' + m[0] + '}$', label)
            text.set_text(formatted)
            # Tick formatters restore text during drawing; update their labels too.
            for ax in fig.axes:
                for axis in [ax.xaxis, ax.yaxis]:
                    labels = axis.get_ticklabels()
                    if any(item is text for item in labels):
                        axis.set_ticklabels([formatted if item is text else item.get_text() for item in labels])
        changes.append(label)
    require(len(old_geometry) == len(geometry(fig)), 'Integrity check failed: len(old_geometry) == len(geometry(fig))')
    for before, after in zip(old_geometry, geometry(fig)):
        require(np.array_equal(before.astype(float), after.astype(float), equal_nan=True), stem)
    new_text = [re.sub(r'\$\\mathit\{([^}]+)\}\$', r'\1', t.get_text()) for t in fig.findobj(Text)]
    require(old_text == new_text, (stem,[(a,b) for a,b in zip(old_text,new_text) if a!=b]))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['png','pdf','svg']:
        kwargs = {'dpi':600 if stem.startswith(('F2_','F3_','F4_','F5_')) else 300}
        if stem.startswith('F'):
            kwargs['bbox_inches'] = 'tight'
        fig.savefig(OUT/f'{stem}.{ext}', **kwargs)
    records.append({'figure':stem,'gene_labels_styled':len(changes),
                    'labels':changes,'numerical_artists_unchanged':True,'visible_text_unchanged':True})
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    # Inherited plot code reads preserved results. Redirect all writes to this revision.
    sys.path.insert(0, str(ROOT/'baseline_114/scripts'))
    original = load('preserved_paired_plots', ROOT/'baseline_114/scripts/make_figures.py')
    original.SRC = QA/'computed_source_data'
    original.SRC.mkdir(exist_ok=True)
    original.save_all_formats = lambda fig, stem: save(fig, stem, records)
    original.manifest = lambda *args, **kwargs: None
    clinical, wide, _ = original.read_inputs()
    paired = original.read_required_csv(original.ROOT/'results/paired.csv')
    original.figure2(wide, paired)
    original.figure3(clinical, wide)
    original.figure4(paired)
    original.figure5()
    # These imports change plotting defaults, so they follow the original figures.
    discovery = load('discovery_typography', ROOT/'scripts/make_discovery_figure.py')
    fig, axes = plt.subplots(2,1,figsize=(6.5,8.3),gridspec_kw={'height_ratios':[1.4,1]},layout='constrained')
    discovery.draw_panel_a(axes[0],discovery.load_json('registry/original_gene_selection.json')['discovery'])
    discovery.draw_panel_b(axes[1],discovery.load_json('baseline_114/results/data_summary.json'),discovery.load_json('baseline_114/registry/ml_config.json'))
    save(fig,'F1_data_flow',records)
    reviewer = load('reviewer_typography', ROOT/'scripts/make_reviewer_figures.py')
    reviewer.save = lambda fig, stem: save(fig, stem, records)
    reviewer.promoter()
    source_checks = []
    for p in original.SRC.glob('*.tsv'):
        old = original.ROOT/'figures/source_data'/p.name
        require(old.is_file() and p.read_bytes() == old.read_bytes(), p.name)
        source_checks.append(p.name)
    artifacts = {}
    before_path = QA/'before_hashes.json'
    before_hashes = json.loads(before_path.read_text()) if before_path.is_file() else {}
    for p in OUT.iterdir():
        if p.suffix not in {'.png','.pdf','.svg'}:
            continue
        folder = 'figures/reviewer' if p.stem.startswith('S') else 'figures'
        for dest in [ROOT/folder/p.name,ROOT/'submission/figures'/p.name]:
            rel = str(dest.relative_to(ROOT))
            artifacts[rel] = {'before_sha256':before_hashes.get(rel,sha(dest)),'after_sha256':sha(p)}
            shutil.copy2(p,dest)
    for row in records:
        stem = row['figure']
        if not stem.startswith('F'):
            continue
        tiff = ROOT/'figures'/f'{stem}.tiff'
        if tiff.is_file():
            rel = str(tiff.relative_to(ROOT))
            old_sha = before_hashes.get(rel,sha(tiff))
            with Image.open(ROOT/'figures'/f'{stem}.png') as im:
                im.save(tiff,compression='tiff_lzw')
            artifacts[rel] = {'before_sha256':old_sha,'after_sha256':sha(tiff)}
        manifest = ROOT/'figures'/f'{stem}_manifest.json'
        if not manifest.is_file():
            continue
        backup = QA/'before/figures'/manifest.name
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():
            shutil.copy2(manifest,backup)
        rel = str(manifest.relative_to(ROOT))
        old_sha = before_hashes.get(rel,sha(backup))
        data = json.loads(manifest.read_text())
        data['typography_revision'] = {'script':'scripts/style_gene_figures.py',
            'script_sha256':sha(Path(__file__)),
            'source_manifest':str(backup.relative_to(ROOT)),
            'numerical_artists_unchanged':True,'visible_text_unchanged':True}
        for output in data['outputs']:
            path = ROOT/output['path']
            output.update(bytes=path.stat().st_size,sha256=sha(path))
            if path.suffix=='.png':
                with Image.open(path) as im:
                    output.update(png_width=im.width,png_height=im.height)
        manifest.write_text(json.dumps(data,indent=2)+'\n')
        artifacts[rel] = {'before_sha256':old_sha,'after_sha256':sha(manifest)}
    (QA/'figure_checks.json').write_text(json.dumps({'status':'pass','figures':records,
        'recomputed_plot_source_tables_identical':source_checks,'artifacts':artifacts},indent=2)+'\n')
    print(json.dumps({'figures_styled':len(records),'source_tables_identical':len(source_checks)}))


if __name__ == '__main__':
    main()
