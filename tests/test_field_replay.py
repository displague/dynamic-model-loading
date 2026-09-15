import numpy as np
import pytest
from dynamic_model_loading.field_replay import validate_completion


@pytest.mark.parametrize('key,value',[('complete',False),('new_inference',True),
    ('input_tensor_bytes',9),('prediction_tensor_bytes',15),('process_peak_rss',10),
    ('calculation_seconds',float('nan')),('calculation_seconds',11)])
def test_resource_receipt_fails_closed(key,value):
    inputs={'x':np.zeros(1)}; predictions={'x':np.zeros(2)}
    row=dict(complete=True,new_inference=False,input_tensor_bytes=8,prediction_tensor_bytes=16,
        process_peak_rss=100,calculation_seconds=1.)
    validate_completion(row,inputs,predictions,dict(worker_wall_seconds=10))
    row[key]=value
    with pytest.raises(ValueError): validate_completion(row,inputs,predictions,dict(worker_wall_seconds=10))
