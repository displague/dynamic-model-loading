"""Render the validated causal development result without changing its decisions."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .experiment import digest, write_json


def render(analysis_path, output_path):
    analysis_path,output=Path(analysis_path),Path(output_path)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'causal_plot.py')
    shutil.copyfile(analysis_path,output/'input-summary.json')
    data=json.loads(analysis_path.read_text(encoding='utf-8'))
    if data['status']!='validated':raise ValueError('Validated analysis required')
    rows=data['selectors']; names=[r['mode'] for r in rows]
    if names!=['static','recency','ema','learned','hindsight']:
        raise ValueError('Unexpected selector grid')
    # Each scored document has exactly one more input than predicted tokens.
    input_tokens=rows[0]['predicted_tokens']+data['row_counts']['dense_reference']-1
    ppl=[r['relative_perplexity'] for r in rows]
    traffic=[r['warm_bytes']/input_tokens/2**30 for r in rows]
    baseline=rows[0]['dense_baseline']['total_bytes']/input_tokens/2**30
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'causal-v070'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    positions=np.arange(len(rows));colors=['#355c7d']*4+['#c27726']
    axes[0].scatter(positions,ppl,c=colors,s=55,zorder=3)
    axes[0].axhline(1.01,color='#a23b3b',linestyle='--',label='Development limit: 1.01')
    low=min(.99,min(ppl)-.01);high=max(1.025,max(ppl)*1.015)
    axes[0].set_ylim(low,high)
    axes[0].set_ylabel('Relative perplexity vs native dense')
    axes[0].set_title('Quality at fixed 90% retention')
    axes[0].legend(loc='upper right',fontsize=8)
    for x,y in zip(positions,ppl,strict=True):
        axes[0].annotate(f'{y:.4f}',(x,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[1].bar(positions,traffic,color=colors,width=.65)
    axes[1].axhline(baseline,color='#444444',linestyle=':',label='Strongest dense static baseline')
    axes[1].axhline(.9*baseline,color='#a23b3b',linestyle='--',label='10% traffic-saving screen')
    axes[1].set_ylabel('Simulated warm FFN traffic (GiB / input token)')
    axes[1].set_title('Selector storage reduces cache capacity')
    axes[1].set_ylim(0,max(baseline,max(traffic))*1.14)
    axes[1].legend(loc='upper left',fontsize=8)
    for x,y in zip(positions,traffic,strict=True):
        axes[1].annotate(f'{y:.3f}',(x,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    costs=[r['cost_ratios'] for r in rows[:4]]
    if any(r is None for r in costs):raise ValueError('Measured CUDA cost required for this figure')
    cuda=[r['cuda_ms']*100 for r in costs];wall=[r['wall_ms']*100 for r in costs]
    axes[2].bar(positions[:4]-.18,cuda,width=.34,color='#355c7d',label='CUDA median')
    axes[2].bar(positions[:4]+.18,wall,width=.34,color='#68a0b0',label='Synchronized wall median')
    axes[2].axhline(10,color='#a23b3b',linestyle='--',label='Affordability screen: 10%')
    axes[2].set_ylim(0,max(12,max(cuda+wall)*1.2))
    axes[2].set_title('Query + necessary observation update')
    axes[2].set_ylabel('Selector / resident dense FFN time (%)')
    axes[2].legend(loc='upper left',fontsize=8)
    for ax in axes:
        labels=['EMA' if name=='ema' else name.title() for name in names]
        ax.set_xticks(positions if ax is not axes[2] else positions[:4],
                      labels if ax is not axes[2] else labels[:4],
                      rotation=20,ha='right')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Qwen causal selection: reused development data, dense diagnostic execution',fontsize=13)
    for suffix in ('svg','png'):
        fig.savefig(output/f'causal-frontier.{suffix}',dpi=180,metadata={'Date':None} if suffix=='svg' else None)
    plt.close(fig)
    write_json(output/'manifest.json',{'analysis_sha256':digest(analysis_path),'plot_source_sha256':digest(Path(__file__)),
        'input_tokens':input_tokens,'artifacts':{p.name:digest(p) for p in output.glob('causal-frontier.*')},
        'note':'Hindsight is noncausal and has no eligible cost bar. Traffic excludes actual transfer execution; no speedup claim.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();render(args.analysis,args.output)


if __name__=='__main__':main()
