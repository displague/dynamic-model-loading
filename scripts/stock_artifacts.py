"""Verify pinned files and fingerprint tokenizer fields without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from stock_benchmark import digest, verify_catalog, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--llama-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    catalogue=root/'configs/stock-speculation-artifacts.json'
    catalog=json.loads(catalogue.read_text(encoding='utf-8'))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.llama_source,text=True).strip()
    if revision!=catalog['runtime']['commit']:
        raise ValueError('unpinned llama.cpp source revision')
    if subprocess.check_output(['git','status','--porcelain'],cwd=args.llama_source,text=True).strip():
        raise ValueError('llama.cpp source worktree is dirty')
    sys.path.insert(0,str(args.llama_source.resolve()/'gguf-py'))
    from gguf import GGUFReader
    checked=verify_catalog(catalog,args.models,args.binary,list(catalog['models']))
    receipt={'catalog_sha256':digest(catalogue),'runtime_source':revision,'models':{}}
    vocabularies={}
    for key,paths in checked.items():
        reader=GGUFReader(paths[0])
        fields={}
        for name,field in reader.fields.items():
            if name.startswith('tokenizer.'):
                value=field.contents()
                encoded=json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf-8')
                fields[name]={'sha256':hashlib.sha256(encoded).hexdigest(),
                              'entries':len(value) if isinstance(value,list) else 1}
                if not isinstance(value,list):fields[name]['value']=value
        vocabularies[key]=reader.fields['tokenizer.ggml.tokens'].contents()
        receipt['models'][key]={'files':[{'path':p,'size':Path(p).stat().st_size,
                                         'mtime_ns':Path(p).stat().st_mtime_ns} for p in paths],
                                'tokenizer':fields,'vocab_size':len(vocabularies[key])}
        del reader
    target=vocabularies['target']
    for key,tokens in vocabularies.items():
        mismatches=[i for i,(a,b) in enumerate(zip(target,tokens)) if a!=b]
        receipt['models'][key]['common_token_id_mismatches']=mismatches
        receipt['models'][key]['vocabulary_length_difference']=len(tokens)-len(target)
    # Metadata fingerprints document compatibility; the stock runtime still performs its own check.
    with args.output.open('x',encoding='utf-8') as out:
        out.write(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print('Verified files and tokenizer fingerprints:',args.output)


if __name__=='__main__':main()
