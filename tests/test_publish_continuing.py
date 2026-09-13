from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_continuing as delivery


def test_publication_rejects_remote_tag_object_or_target_drift():
    good='object refs/tags/v0.16.0\nhead refs/tags/v0.16.0^{}'
    delivery.verify_remote_tag(good,'object','head')
    for bad in [good.replace('object','stale'),good.replace('head','stale'),
                'head refs/tags/v0.16.0',good+'\nhead refs/tags/v0.16.0^{}']:
        with pytest.raises(ValueError,match='remote annotated tag identity'):
            delivery.verify_remote_tag(bad,'object','head')
