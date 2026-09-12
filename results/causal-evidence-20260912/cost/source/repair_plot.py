"""Plot all completed fixed-mask repair curves without selecting a runtime policy."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .experiment import digest, write_json


def render(run,analysis,output):
    run,analysis,output=Path(run),Path(analysis),Path(output)
    verified=json.loads((analysis/'summary.json').read_text(encoding='utf-8'))
    if verified['status']!='validated' or verified['input_hashes']['results.jsonl']!=digest(run/'results.jsonl'):
        raise ValueError('Validated matching repair ledger required')
    rows=[json.loads(line) for line in (run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'repair_plot.py')
    conditions=[]
    for aggregate in verified['conditions']:
        documents=[r for r in rows if r['kind'] in ('repair_document','one_shot_document') and r['condition']==aggregate['condition']]
        first=documents[0];repair=first['kind']=='repair_document';visits=sum(r['predicted_tokens']+1 for r in documents)
        total=visits*28*1120
        selected=sum(r['final_selected_bytes'] if repair else r['replay'][0]['selected_request_bytes'] for r in documents)
        value={**aggregate,'mode':first['mode'],'schedule':first['schedule'],
            'limit':first.get('limit'),'keep':first.get('keep'),'individual_quality_passes':sum(r['relative_perplexity']<=1.01 for r in documents),
            'document_relative_perplexities':[r['relative_perplexity'] for r in documents],
            'warm_gib_per_input':aggregate['warm_bytes']/visits/2**30,
            'selected_ffn_fraction':selected/(total*147456)}
        if repair:
            for key in ('added_groups','resident_added_groups','cold_added_groups'):
                value[key+'_per_layer_visit']=sum(r[key] for r in documents)/(visits*28)
            value['added_cold_gib_per_input']=sum(r['added_cold_bytes'] for r in documents)/visits/2**30
        conditions.append(value)
    write_json(output/'curves.json',conditions)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,mode in zip(axes.flat,('static','recency','ema','learned'),strict=True):
        for schedule,style,color in (('hindsight','o-','#2563eb'),('resident_first','s-','#b45309'),('one_shot','^--','#64748b')):
            subset=[r for r in conditions if r['mode']==mode and r['schedule']==schedule]
            ax.plot([100*r['warm_saving'] for r in subset],[r['relative_perplexity'] for r in subset],style,color=color,label=schedule.replace('_',' '),markersize=5)
        ax.axvline(10,color='#166534',lw=1);ax.axhline(1.01,color='#166534',lw=1)
        ax.fill_between([10,20],.99,1.01,color='#dcfce7',alpha=.7)
        ax.set(title=mode.capitalize(),xlim=(-2,20),ylim=(.99,1.19));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.supxlabel('Simulated warm transfer saving versus dense static cache (%)')
    fig.supylabel('Perplexity relative to native dense')
    fig.suptitle('Fixed-mask repair: all 68 aggregate conditions\nTwo reused article prefixes; privileged curves cannot nominate a runtime',fontsize=13)
    fig.tight_layout();fig.savefig(output/'repair-curves.svg');fig.savefig(output/'repair-curves.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','analysis','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();render(args.run,args.analysis,args.output)


if __name__=='__main__':main()
