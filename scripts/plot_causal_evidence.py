"""Render every frozen quality/cost point; no fitted trend or policy selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();summary=json.loads(Path(args.analysis).read_text(encoding='utf-8'))
    if summary['status']!='verified':raise ValueError('Verified analysis required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=summary['conditions'];points=[]
    for row in rows:
        if row['quality'] is None:raise ValueError('A plot cannot silently drop an invalid quality point')
        points.append(dict(mode=row['mode'],extra=row['extra'],relative_ppl=row['quality']['relative_perplexity'],
            warm_gib_per_input=row['warm_bytes']/512/2**30,
            serialized_ms_per_input=row['serialized_cost_ms']['total_ms']/512,
            prefix_passes=row['individual_quality_passes'],gate_b=row['gate_b_wiki'],gate_c=row['gate_c_plausibility']))
    dense_bytes=rows[0]['warm_bytes']/(1-rows[0]['warm_saving'])/512/2**30
    dense_ms=rows[0]['serialized_dense_ms']/512
    palette={'initial':('#222222','x','Initial, no repair'),'one_shot':('#2475b0','s','Larger one-shot'),
        'predetermined':('#777777','^','Predetermined additions'),'partial':('#db7d12','o','Partial evidence'),
        'input_only':('#8e5ab5','D','Input-only ablation'),'fallback':('#b94343','P','Detector / fallback'),
        'full':('#172d41','*','Full completion')}
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,6.2),sharey=True)
        for mode,(color,marker,label) in palette.items():
            chosen=sorted((p for p in points if p['mode']==mode),key=lambda p:p['extra'])
            for ax,xkey in zip(axes,('warm_gib_per_input','serialized_ms_per_input')):
                ax.plot([p[xkey] for p in chosen],[p['relative_ppl'] for p in chosen],
                    color=color,marker=marker,markersize=8,linewidth=1.4,label=label)
                if len(chosen)>1:
                    for p in chosen:ax.annotate(str(p['extra']), (p[xkey],p['relative_ppl']),xytext=(5,6),textcoords='offset points',fontsize=8,color=color)
        for ax in axes:
            ax.axhline(1.01,color='#a52e31',linestyle='--',linewidth=1,label='PPL gate: 1.01')
            ax.axhline(1.,color='#222222',linewidth=.7,alpha=.5)
            ax.grid(axis='y',alpha=.22)
        axes[0].axvline(.9*dense_bytes,color='#456b46',linestyle=':',linewidth=1.4)
        axes[1].axvline(dense_ms,color='#456b46',linestyle=':',linewidth=1.4)
        axes[0].set(xlabel='Warm FFN link traffic (GiB / input token)',ylabel='Relative perplexity vs dense',
                    title='Quality versus acquired weights + dispatch')
        axes[1].set(xlabel='Serialized primitive estimate (ms / input token)',title='Quality versus charged action cost')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.5,.005))
        fig.suptitle('Causal repair on four reused Wiki prefixes',fontsize=14)
        fig.text(.5,.125,'Dotted lines: 10% traffic-saving target (left), dense serialized comparator (right). Numbers denote extra groups.',ha='center',fontsize=9)
        fig.subplots_adjust(bottom=.29,top=.84,wspace=.16,left=.07,right=.985)
        fig.savefig(output/'causal-evidence-quality-cost.png',dpi=160,bbox_inches='tight')
        fig.savefig(output/'causal-evidence-quality-cost.svg',bbox_inches='tight')
        plt.close(fig)
    (output/'causal-evidence-points.json').write_text(json.dumps({'points':points,'dense_gib_per_input':dense_bytes,
        'dense_serialized_ms_per_input':dense_ms,'note':'Synthetic serialized costs are not measured inference latency.'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
